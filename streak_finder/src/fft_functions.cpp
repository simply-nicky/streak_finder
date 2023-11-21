#include "fft_functions.hpp"

namespace streak_finder {

size_t next_fast_len(size_t target)
{
    if (target <= 16) return target;
    if (!(target & (target - 1))) return target;
    if (target <= detail::LPRE[detail::LPRE.size() - 1])
    {
        return *searchsorted(target, detail::LPRE.cbegin(), detail::LPRE.cend(), side::left, std::less<size_t>());
    }
    size_t match, best_match = 2 * target;

    match = detail::find_match(target, 1);
    if (match < best_match) best_match = match;
    match = detail::find_match(target, 11);
    if (match < best_match) best_match = match;
    match = detail::find_match(target, 13);
    if (match < best_match) best_match = match;
    return best_match;
}

template <typename Inp, typename Krn, typename Seq>
auto fft_convolve(py::array_t<Inp> inp, py::array_t<Krn> kernel, std::optional<Seq> axis, unsigned threads)
{
    using Out = std::common_type_t<Inp, Krn>;
    assert(PyArray_API);

    sequence<long> seq;
    if (!axis)
    {
        if (inp.ndim() != kernel.ndim())
            throw std::invalid_argument("inp and kernel have different numbers of dimensions: " +
                                        std::to_string(inp.ndim()) + " and " + std::to_string(kernel.ndim()));

        std::vector<long> vec (inp.ndim(), 0);
        std::iota(vec.begin(), vec.end(), 0);
        seq = std::move(vec);
    }
    else seq = axis.value();

    if (seq.size() != static_cast<size_t>(kernel.ndim()))
        throw std::invalid_argument("Wrong number of axes (" + std::to_string(seq.size()) +
                                    "), must be " + std::to_string(kernel.ndim()));

    seq = seq.unwrap(inp.ndim());
    inp = seq.swap_axes(inp);

    auto iarr = array<Inp>(inp.request());
    auto karr = array<Krn>(kernel.request());
    auto out = py::array_t<Out>(iarr.shape);
    auto oarr = array<Out>(out.request());

    auto ax = iarr.ndim - seq.size();
    std::vector<size_t> axes (seq.size());
    std::iota(axes.begin(), axes.end(), ax);
    auto repeats = get_size(iarr.shape.begin(), std::next(iarr.shape.begin(), ax));
    threads = (threads > repeats) ? repeats : threads;

    std::vector<size_t> fshape;
    std::transform(karr.shape.begin(), karr.shape.end(), std::next(iarr.shape.begin(), ax),
                   std::back_inserter(fshape), [](size_t nk, size_t ni){return next_fast_len(nk + ni);});
    auto bshape = fftw_buffer_shape<Out>(fshape);

    Out factor = 1.0 / get_size(fshape.begin(), fshape.end());

    thread_exception e;

    py::gil_scoped_release release;

    vector_array<Out> kbuffer (bshape);
    write_buffer(kbuffer, fshape, std::move(karr));
    auto kbuf_inp = kbuffer.ptr;
    auto kbuf_out = reinterpret_cast<std::complex<remove_complex_t<Out>> *>(kbuffer.ptr);

    auto fwd_plan = make_forward_plan(fshape, kbuf_inp, kbuf_out);
    auto bwd_plan = make_backward_plan(fshape, kbuf_out, kbuf_inp);

    fftw_execute(fwd_plan, kbuf_inp, kbuf_out);

    #pragma omp parallel num_threads(threads)
    {
        vector_array<Out> ibuffer (bshape);

        auto ibuf_inp = ibuffer.ptr;
        auto ibuf_out = reinterpret_cast<std::complex<remove_complex_t<Out>> *>(ibuffer.ptr);
        auto buf_size = is_complex_v<Out> ? ibuffer.size : ibuffer.size / 2;

        #pragma omp for
        for (size_t i = 0; i < repeats; i++)
        {
            e.run([&]
            {
                write_buffer(ibuffer, fshape, iarr.slice(i, axes));
                fftw_execute(fwd_plan, ibuf_inp, ibuf_out);
                for (size_t j = 0; j < buf_size; j++) ibuf_out[j] *= kbuf_out[j] * factor;
                fftw_execute(bwd_plan, ibuf_out, ibuf_inp);
                read_buffer(ibuffer, fshape, oarr.slice(i, axes));
            });
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return seq.swap_axes_back(out);
}

}

PYBIND11_MODULE(fft_functions, m)
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

    m.def("fft_convolve", &fft_convolve<float, float, int>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<float, float, std::vector<int>>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<double, double, int>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<double, double, std::vector<int>>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);

    m.def("fft_convolve", &fft_convolve<std::complex<float>, float, int>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<std::complex<float>, float, std::vector<int>>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<std::complex<double>, double, int>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<std::complex<double>, double, std::vector<int>>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);

    m.def("fft_convolve", &fft_convolve<float, std::complex<float>, int>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<float, std::complex<float>, std::vector<int>>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<double, std::complex<double>, int>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<double, std::complex<double>, std::vector<int>>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);

    m.def("fft_convolve", &fft_convolve<std::complex<float>, std::complex<float>, int>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<std::complex<float>, std::complex<float>, std::vector<int>>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<std::complex<double>, std::complex<double>, int>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<std::complex<double>, std::complex<double>, std::vector<int>>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);

}