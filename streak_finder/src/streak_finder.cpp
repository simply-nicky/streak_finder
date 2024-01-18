#include "streak_finder.hpp"

namespace streak_finder {

template <typename T>
std::vector<std::array<T, 4>> detect_streaks(Peaks peaks, py::array_t<T> data, py::array_t<bool> mask, Structure structure,
                                             T xtol, T vmin, T log_eps, unsigned max_iter, unsigned lookahead, size_t min_size)
{
    Pattern<T> pattern {data, mask, std::move(structure)};
    return pattern.find_streaks(std::move(peaks), xtol, vmin, log_eps, max_iter, lookahead, min_size);
}

}

PYBIND11_MODULE(streak_finder, m)
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

    py::class_<Structure>(m, "Structure")
        .def(py::init<int, int>(), py::arg("radius"), py::arg("rank"))
        .def_readonly("radius", &Structure::radius)
        .def_readonly("rank", &Structure::rank)
        .def_property("size", [](Structure & srt){return srt.idxs.size();}, nullptr)
        .def_property("x", [](Structure & srt){return detail::get_x(srt.idxs);}, nullptr)
        .def_property("y", [](Structure & srt){return detail::get_y(srt.idxs);}, nullptr)
        .def("__repr__", &Structure::info);

    py::class_<Peaks>(m, "Peaks")
        .def(py::init<py::array_t<float>, size_t, float>(), py::arg("data"), py::arg("radius"), py::arg("vmin"))
        .def(py::init<py::array_t<double>, size_t, double>(), py::arg("data"), py::arg("radius"), py::arg("vmin"))
        .def("filter",
            [](Peaks & peaks, py::array_t<float> data, Structure s, float vmin, size_t npts)
            {
                peaks.filter(array<float>(data.request()), s, vmin, npts);
                return peaks;
            }, py::arg("data"), py::arg("structure"), py::arg("vmin"), py::arg("npts"))
        .def("filter",
            [](Peaks & peaks, py::array_t<double> data, Structure s, double vmin, size_t npts)
            {
                peaks.filter(array<double>(data.request()), s, vmin, npts);
                return peaks;
            }, py::arg("data"), py::arg("structure"), py::arg("vmin"), py::arg("npts"))
        .def_property("size", [](Peaks & peaks){return peaks.points.size();}, nullptr)
        .def_property("x", [](Peaks & peaks){return detail::get_x(peaks.points);}, nullptr)
        .def_property("y", [](Peaks & peaks){return detail::get_y(peaks.points);}, nullptr)
        .def("__repr__", &Peaks::info);

    m.def("detect_streaks", &detect_streaks<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("log_eps")=std::log(1e-1), py::arg("max_iter")=100, py::arg("lookahead")=1, py::arg("min_size")=5);
    m.def("detect_streaks", &detect_streaks<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("log_eps")=std::log(1e-1), py::arg("max_iter")=100, py::arg("lookahead")=1, py::arg("min_size")=5);
}