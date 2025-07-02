#include "bresenham.hpp"

namespace streak_finder {

template <typename T>
using kernel_t = typename kernels<T>::kernel;

template <typename T>
using py_array_t = typename py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename I, int ExtraFlags>
void fill_indices(std::string name, size_t xsize, size_t isize, std::optional<py::array_t<I, ExtraFlags>> & idxs)
{
    idxs = py::array_t<I, ExtraFlags>(isize);

    if (isize)
    {
        if (xsize == 1) fill_array(idxs.value(), I());
        else if (xsize == isize)
        {
            for (size_t i = 0; i < isize; i++) idxs.value().mutable_data()[i] = i;
        }
        else throw std::invalid_argument(name + " has an icnompatible size (" + std::to_string(isize) + " != " +
                                         std::to_string(xsize) + ")");
    }
}

template <typename I, int ExtraFlags>
void check_indices(std::string name, size_t imax, size_t isize, const py::array_t<I, ExtraFlags> & idxs)
{
    if (idxs.size())
    {
        if (static_cast<size_t>(idxs.size()) != isize)
            throw std::invalid_argument(name + " has an invalid size (" + std::to_string(idxs.size()) +
                                        " != " + std::to_string(isize) + ")");

        auto [min, max] = std::minmax_element(idxs.data(), idxs.data() + idxs.size());
        if (*max >= static_cast<I>(imax) || *min < I())
            throw std::out_of_range(name + " range (" + std::to_string(*min) + ", " + std::to_string(*max) +
                                    ") is outside of (0, " + std::to_string(imax) + ")");
    }
}

template <typename T, typename I, size_t N, int Update>
py::array_t<T> draw_lines_nd(py_array_t<T> out, py_array_t<T> lines, std::optional<py_array_t<I>> idxs, T max_val,
                             std::string kernel, unsigned threads)
{
    assert(PyArray_API);
    constexpr static size_t L = 2 * N + 1;

    auto krn = kernels<T>::get_kernel(kernel);
    array<T> oarr {out.request()};
    array<T> larr (lines.request());

    auto n_frames = std::reduce(oarr.shape().begin(), std::prev(oarr.shape().end(), N), size_t(1), std::multiplies());
    std::vector<size_t> shape {std::prev(oarr.shape().end(), N), oarr.shape().end()};

    check_dimensions("lines", larr.ndim() - 1, larr.shape(), L);
    auto lsize = larr.size() / larr.shape(larr.ndim() - 1);

    if (!idxs) fill_indices("idxs", n_frames, lsize, idxs);
    else check_indices("idxs", n_frames, lsize, idxs.value());
    auto iarr = array<I>(idxs.value().request());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        detail::ImageBuffer<size_t, T> buffer (shape);

        auto write = [&oarr, size = buffer.size()](const std::tuple<size_t, size_t, T> & values)
        {
            auto [idx, frame, value] = values;
            size_t index = idx + size * frame;

            if constexpr (Update) oarr[index] = std::max(oarr[index], value);
            else oarr[index] = oarr[index] + value;
        };

        #pragma omp for nowait
        for (size_t i = 0; i < lsize; i++)
        {
            e.run([&]()
            {
                auto draw_pixel = [&buffer, &krn, max_val, frame = iarr[i]](const PointND<long, N> & pt, T error)
                {
                    if (error <= 1.0 && buffer.is_inbound(pt.rbegin(), pt.rend()))
                    {
                        buffer.emplace_back(pt, frame, max_val * krn(std::sqrt(error)));
                    }
                };

                draw_line_nd(LineND<T, N>{to_point<N>(larr, L * i), to_point<N>(larr, L * i + N)}, larr[L * i + 2 * N], draw_pixel);
            });
        }

        #pragma omp critical
        std::for_each(buffer.begin(), buffer.end(), write);
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T, typename I, int Update>
py::array_t<T> draw_lines_2d_3d(py_array_t<T> out, py_array_t<T> lines, std::optional<py_array_t<I>> idxs, T max_val,
                                std::string kernel, unsigned threads)
{
    size_t L = lines.shape(lines.ndim() - 1);
    switch (L)
    {
        case 5:
            return draw_lines_nd<T, I, 2, Update>(out, lines, idxs, max_val, kernel, threads);
        case 7:
            return draw_lines_nd<T, I, 3, Update>(out, lines, idxs, max_val, kernel, threads);
        default:
            throw std::runtime_error("Invalid lines size (" + std::to_string(L) + ") at axis " +
                                     std::to_string(lines.ndim() - 1));
    }
}

template <typename T, typename I>
py::array_t<T> draw_lines(py_array_t<T> out, py_array_t<T> lines, std::optional<py_array_t<I>> idxs, T max_val,
                          std::string kernel, std::string overlap, unsigned threads)
{
    if (overlap == "sum") return draw_lines_2d_3d<T, I, 0>(out, lines, idxs, max_val, kernel, threads);
    if (overlap == "max") return draw_lines_2d_3d<T, I, 1>(out, lines, idxs, max_val, kernel, threads);
    throw std::invalid_argument("Invalid overlap keyword: " + overlap);
}

template <typename T, typename I, size_t N>
auto write_lines_nd(py_array_t<T> lines, std::vector<size_t> shape, std::optional<py_array_t<I>> idxs,
                    T max_val, std::string kernel, unsigned threads)
{
    assert(PyArray_API);
    constexpr static size_t L = 2 * N + 1;

    auto krn = kernels<T>::get_kernel(kernel);

    auto n_frames = std::reduce(shape.begin(), std::prev(shape.end(), N), size_t(1), std::multiplies());
    std::vector<size_t> fshape {std::prev(shape.end(), N), shape.end()};

    auto larr = array<T>(lines.request());
    check_dimensions("lines", larr.ndim() - 1, larr.shape(), L);
    auto lsize = larr.size() / larr.shape(larr.ndim() - 1);

    if (!idxs) fill_indices("idxs", n_frames, lsize, idxs);
    else check_indices("idxs", n_frames, lsize, idxs.value());
    auto iarr = array<I>(idxs.value().request());

    std::vector<I> out_idxs, lidxs;
    std::vector<T> values;

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        detail::ImageBuffer<size_t, size_t, T> buffer {fshape};

        auto merge = [&out_idxs, &lidxs, &values, size = buffer.size()](const std::tuple<size_t, size_t, size_t, T> & value)
        {
            out_idxs.push_back(std::get<0>(value) + size * std::get<1>(value));
            lidxs.push_back(std::get<2>(value));
            values.push_back(std::get<3>(value));
        };

        #pragma omp for nowait
        for (size_t i = 0; i < lsize; i++)
        {
            e.run([&]()
            {
                auto draw_pixel = [&buffer, &krn, max_val, frame = iarr[i], id = i](const PointND<long, N> & pt, T error)
                {
                    if (error <= 1.0 && buffer.is_inbound(pt.rbegin(), pt.rend()))
                    {
                        buffer.emplace_back(pt, frame, id, max_val * krn(std::sqrt(error)));
                    }
                };

                draw_line_nd(LineND<T, N>{to_point<N>(larr, L * i), to_point<N>(larr, L * i + N)}, larr[L * i + 2 * N], draw_pixel);
            });
        }

        #pragma omp critical
        std::for_each(buffer.begin(), buffer.end(), merge);
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return std::make_tuple(as_pyarray(std::move(out_idxs)), as_pyarray(std::move(lidxs)), as_pyarray(std::move(values)));
}

template <typename T, typename I>
auto write_lines(py_array_t<T> lines, std::vector<size_t> shape, std::optional<py_array_t<I>> idxs,
                 T max_val, std::string kernel, unsigned threads)
{
    size_t L = lines.shape(lines.ndim() - 1);
    switch (L)
    {
        case 5:
            return write_lines_nd<T, I, 2>(lines, shape, idxs, max_val, kernel, threads);
        case 7:
            return write_lines_nd<T, I, 3>(lines, shape, idxs, max_val, kernel, threads);
        default:
            throw std::runtime_error("Invalid lines size (" + std::to_string(L) + ") at axis " +
                                     std::to_string(lines.ndim() - 1));
    }
}

}

PYBIND11_MODULE(bresenham, m)
{
    using namespace streak_finder;

    try
    {
        import_numpy();
    }
    catch (const py::error_already_set & e)
    {
        return;
    }

    m.def("draw_lines",
        [](py_array_t<double> lines, std::vector<size_t> shape, std::optional<py_array_t<size_t>> idxs, double max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py_array_t<double> out {shape};
            fill_array(out, double());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
    m.def("draw_lines",
        [](py_array_t<float> lines, std::vector<size_t> shape, std::optional<py_array_t<size_t>> idxs, float max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py_array_t<float> out {shape};
            fill_array(out, float());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
    m.def("draw_lines",
        [](py_array_t<double> lines, std::vector<size_t> shape, std::optional<py_array_t<long>> idxs, double max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py_array_t<double> out {shape};
            fill_array(out, double());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
    m.def("draw_lines",
        [](py_array_t<float> lines, std::vector<size_t> shape, std::optional<py_array_t<long>> idxs, float max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py_array_t<float> out {shape};
            fill_array(out, float());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);

    m.def("write_lines", &write_lines<float, size_t>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("write_lines", &write_lines<double, size_t>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("write_lines", &write_lines<float, long>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("write_lines", &write_lines<double, long>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
}
