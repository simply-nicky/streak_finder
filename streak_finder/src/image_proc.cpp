#include "image_proc.hpp"

namespace streak_finder {

template <typename T>
void check_lines(array<T> & lines)
{
    if (lines.ndim == 1 && lines.size == 5) lines = array<T>({1, 5}, lines.ptr);
    check_dimensions("lines", lines.ndim - 2, lines.shape, lines.size / 5, 5);
}

template <typename Function>
void check_shape(const std::vector<size_t> & shape, Function && func)
{
    if (std::forward<Function>(func)(shape) || !get_size(shape.begin(), shape.end()))
    {
        std::ostringstream oss;
        std::copy(shape.begin(), shape.end(), std::experimental::make_ostream_joiner(oss, ", "));
        throw std::invalid_argument("invalid shape: {" + oss.str() + "}");
    }
}

template <typename T, typename Out>
py::array_t<Out> draw_line(py::array_t<T, py::array::c_style | py::array::forcecast> lines,
                           std::vector<size_t> shape, Out max_val, T dilation, std::string prof, unsigned threads)
{
    assert(PyArray_API);

    auto p = profiles<T>::get_profile(prof);

    check_shape(shape, [](const std::vector<size_t> & shape){return shape.size() != 2;});

    auto larr = array<T>(lines.request());
    check_lines(larr);

    auto out = py::array_t<Out>(shape);
    PyArray_FILLWBYTE(out.ptr(), 0);

    auto oarr = array<Out>(out.request());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (size_t j = 0; j < larr.shape[0]; j++)
    {
        e.run([&]
        {
            auto liter = larr.line_begin(1, j);
            draw_bresenham(oarr, &oarr.shape, {liter[0], liter[1], liter[2], liter[3]}, liter[4] + dilation, max_val, p);
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T, typename Out>
py::array_t<Out> draw_line_vec(std::vector<py::array_t<T, py::array::c_style | py::array::forcecast>> lines,
                               std::vector<size_t> shape, Out max_val, T dilation, std::string prof, unsigned threads)
{
    assert(PyArray_API);

    auto p = profiles<T>::get_profile(prof);

    check_shape(shape, [](const std::vector<size_t> & shape){return shape.size() < 3;});

    if (get_size(shape.begin(), std::prev(shape.end(), 2)) != lines.size())
        throw std::invalid_argument("shape is incompatible with the list of lines");

    std::vector<array<T>> lvec;
    for (const auto & obj : lines)
    {
        auto & arr = lvec.emplace_back(obj.request());
        check_lines(arr);
    }

    auto out = py::array_t<Out>(shape);
    PyArray_FILLWBYTE(out.ptr(), 0);

    auto oarr = array<Out>(out.request());
    std::array<size_t, 2> axes = {oarr.ndim - 2, oarr.ndim - 1};

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (size_t i = 0; i < lvec.size(); i++)
    {
        e.run([&]
        {
            auto frame = oarr.slice(i, axes);

            for (size_t j = 0; j < lvec[i].shape[0]; j++)
            {
                auto liter = lvec[i].line_begin(1, j);
                draw_bresenham(frame, &frame.shape, {liter[0], liter[1], liter[2], liter[3]}, liter[4] + dilation, max_val, p);
            }
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T, typename Out>
auto draw_line_table(py::array_t<T, py::array::c_style | py::array::forcecast> lines, std::optional<std::vector<size_t>> shape,
                     Out max_val, T dilation, std::string prof, unsigned threads)
{
    assert(PyArray_API);

    auto p = profiles<T>::get_profile(prof);

    if (shape) check_shape(shape.value(), [](const std::vector<size_t> & shape){return shape.size() != 2;});

    auto larr = array<T>(lines.request());
    check_lines(larr);

    auto sptr = (shape) ? &shape.value() : nullptr;

    table_t<Out> result;

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (size_t j = 0; j < larr.shape[0]; j++)
    {
        e.run([&]
        {
            auto liter = larr.line_begin(1, j);
            draw_bresenham(result, sptr, {liter[0], liter[1], liter[2], liter[3]}, liter[4] + dilation, max_val, p);
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    auto & [xs, ys, vals] = result;

    return std::make_tuple(as_pyarray(std::move(xs)), as_pyarray(std::move(ys)), as_pyarray(std::move(vals)));
}

template <typename T, typename Out>
auto draw_line_table_vec(std::vector<py::array_t<T, py::array::c_style | py::array::forcecast>> lines,
                         std::optional<std::vector<size_t>> shape, Out max_val, T dilation, std::string prof, unsigned threads)
{
    assert(PyArray_API);

    auto p = profiles<T>::get_profile(prof);

    if (shape) check_shape(shape.value(), [](const std::vector<size_t> & shape){return shape.size() != 2;});

    std::vector<array<T>> lvec;
    for (const auto & obj : lines)
    {
        auto & arr = lvec.emplace_back(obj.request());
        check_lines(arr);
    }

    auto sptr = (shape) ? &shape.value() : nullptr;

    std::vector<table_t<Out>> result;

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<table_t<Out>> buffer;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < lvec.size(); i++)
        {
            e.run([&]
            {
                auto & table = buffer.emplace_back();

                for (size_t j = 0; j < lvec[i].shape[0]; j++)
                {
                    auto liter = lvec[i].line_begin(1, j);
                    draw_bresenham(table, sptr, {liter[0], liter[1], liter[2], liter[3]}, liter[4] + dilation, max_val, p);
                }
            });
        }

        #pragma omp for schedule(static) ordered
        for (unsigned i = 0; i < threads; i++)
        {
            #pragma omp ordered
            result.insert(result.end(), std::make_move_iterator(buffer.begin()), std::make_move_iterator(buffer.end()));
        }

    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    std::vector<std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<Out>>> out;
    for (auto & [xs, ys, vals] : result)
    {
        out.emplace_back(as_pyarray(std::move(xs)), as_pyarray(std::move(ys)), as_pyarray(std::move(vals)));
    }

    return out;
}

template <typename T, typename U>
py::array_t<T> local_maxima(py::array_t<T, py::array::c_style | py::array::forcecast> inp, U axis, unsigned threads)
{
    using iterator = typename array<T>::iterator;
    auto ibuf = inp.request();

    sequence<long> seq (axis);
    seq.unwrap(ibuf.ndim);

    for (auto ax : seq)
    {
        if (ibuf.shape[ax] < 3)
            throw std::invalid_argument("The shape along axis " + std::to_string(ax) + "is below 3 (" +
                                        std::to_string(ibuf.shape[ax]) + ")");
    }

    auto iarr = array<T>(ibuf);
    size_t repeats = iarr.size / iarr.shape[seq[0]];
    
    std::vector<T> peaks;

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<size_t> buffer;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < repeats; i++)
        {
            e.run([&]
            {
                // First element can't be a maximum
                auto iter = std::next(iarr.line_begin(seq[0], i));
                auto last = std::prev(iarr.line_end(seq[0], i));
                while (iter != last)
                {
                    if (*std::prev(iter) < *iter)
                    {
                        // ahead can be last
                        auto ahead = std::next(iter);

                        while (ahead != last && *ahead == *iter) ++ahead;

                        if (*ahead < *iter)
                        {
                            std::vector<size_t> peak;
                            auto index = std::distance(iarr.begin(), static_cast<iterator>(iter));
                            iarr.unravel_index(std::back_inserter(peak), index);

                            size_t n = 1;
                            for (; n < seq.size(); n++)
                            {
                                if (peak[seq[n]] > 1 && peak[seq[n]] < iarr.shape[seq[n]] - 1)
                                {
                                    if (iarr[index - iarr.stride(seq[n])] < *iter && iarr[index + iarr.stride(seq[n])] < *iter)
                                    {
                                        continue;
                                    }
                                }

                                break;
                            }

                            if (n == seq.size()) buffer.insert(buffer.end(), std::make_move_iterator(peak.begin()), std::make_move_iterator(peak.end()));

                            // Skip samples that can't be maximum, check if it's not last
                            if (ahead != last) iter = ahead;
                        }
                    }

                    iter = std::next(iter);
                }
            });
        }

        #pragma omp for schedule(static) ordered
        for (unsigned i = 0; i < threads; i++)
        {
            #pragma omp ordered
            peaks.insert(peaks.end(), std::make_move_iterator(buffer.begin()), std::make_move_iterator(buffer.end()));
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    if (peaks.size() % iarr.ndim)
        throw std::runtime_error("peaks have invalid size of " + std::to_string(peaks.size()));

    std::array<size_t, 2> out_shape = {peaks.size() / iarr.ndim, iarr.ndim};
    return as_pyarray(std::move(peaks)).reshape(out_shape);
}

}

PYBIND11_MODULE(image_proc, m)
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

    m.def("draw_line_mask", &draw_line<double, uint32_t>, py::arg("lines"), py::arg("shape"), py::arg("max_val") = 1, py::arg("dilation") = 0.0, py::arg("profile") = "tophat", py::arg("num_threads") = 1);
    m.def("draw_line_mask", &draw_line_vec<double, uint32_t>, py::arg("lines"), py::arg("shape"), py::arg("max_val") = 1, py::arg("dilation") = 0.0, py::arg("profile") = "tophat", py::arg("num_threads") = 1);
    m.def("draw_line_mask", &draw_line<float, uint32_t>, py::arg("lines"), py::arg("shape"), py::arg("max_val") = 1, py::arg("dilation") = 0.0, py::arg("profile") = "tophat", py::arg("num_threads") = 1);
    m.def("draw_line_mask", &draw_line_vec<float, uint32_t>, py::arg("lines"), py::arg("shape"), py::arg("max_val") = 1, py::arg("dilation") = 0.0, py::arg("profile") = "tophat", py::arg("num_threads") = 1);
    
    m.def("draw_line_image", &draw_line<float, float>, py::arg("lines"), py::arg("shape"), py::arg("max_val") = 1.0, py::arg("dilation") = 0.0, py::arg("profile") = "tophat", py::arg("num_threads") = 1);
    m.def("draw_line_image", &draw_line_vec<float, float>, py::arg("lines"), py::arg("shape"), py::arg("max_val") = 1.0, py::arg("dilation") = 0.0, py::arg("profile") = "tophat", py::arg("num_threads") = 1);
    m.def("draw_line_image", &draw_line<double, double>, py::arg("lines"), py::arg("shape"), py::arg("max_val") = 1.0, py::arg("dilation") = 0.0, py::arg("profile") = "tophat", py::arg("num_threads") = 1);
    m.def("draw_line_image", &draw_line_vec<double, double>, py::arg("lines"), py::arg("shape"), py::arg("max_val") = 1.0, py::arg("dilation") = 0.0, py::arg("profile") = "tophat", py::arg("num_threads") = 1);

    m.def("draw_line_table", &draw_line_table<float, float>, py::arg("lines"), py::arg("shape") = std::nullopt, py::arg("max_val") = 1.0, py::arg("dilation") = 0.0, py::arg("profile") = "tophat", py::arg("num_threads") = 1);
    m.def("draw_line_table", &draw_line_table_vec<float, float>, py::arg("lines"), py::arg("shape") = std::nullopt, py::arg("max_val") = 1.0, py::arg("dilation") = 0.0, py::arg("profile") = "tophat", py::arg("num_threads") = 1);
    m.def("draw_line_table", &draw_line_table<double, double>, py::arg("lines"), py::arg("shape") = std::nullopt, py::arg("max_val") = 1.0, py::arg("dilation") = 0.0, py::arg("profile") = "tophat", py::arg("num_threads") = 1);
    m.def("draw_line_table", &draw_line_table_vec<double, double>, py::arg("lines"), py::arg("shape") = std::nullopt, py::arg("max_val") = 1.0, py::arg("dilation") = 0.0, py::arg("profile") = "tophat", py::arg("num_threads") = 1);

    m.def("local_maxima", &local_maxima<int, int>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<int, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<long, int>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<long, int>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<unsigned, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<unsigned, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<size_t, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<size_t, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<float, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<float, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<double, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<double, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);

}