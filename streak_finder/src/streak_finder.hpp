#ifndef STREAK_FINDER_
#define STREAK_FINDER_
#include "array.hpp"
#include "image_proc.hpp"

namespace streak_finder {

namespace detail{

// Taken from the boost::hash_combine: https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
template <typename T, size_t N>
struct ArrayHasher
{
    std::size_t operator()(const std::array<T, N> & arr) const
    {
        std::size_t h = 0;

        for (auto elem : arr)
        {
            h ^= std::hash<T>{}(elem) + 0x9e3779b9 + (h << 6) + (h >> 2); 
        }
        return h;
    }   
};

template <typename T>
struct PointHasher
{
    std::size_t operator()(const Point<T> & point) const
    {
        std::size_t h = 0;

        h ^= std::hash<T>{}(point.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<T>{}(point.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

template <typename Container, typename = std::enable_if_t<is_point_v<typename Container::value_type>>>
auto get_x(const Container & list)
{
    using T = Container::value_type::value_type;
    std::vector<T> x;
    std::transform(list.begin(), list.end(), std::back_inserter(x), [](const Point<T> & elem){return elem.x;});
    return x;
}

template <typename Container, typename = std::enable_if_t<is_point_v<typename Container::value_type>>>
auto get_y(const Container & list)
{
    using T = Container::value_type::value_type;
    std::vector<T> y;
    std::transform(list.begin(), list.end(), std::back_inserter(y), [](const Point<T> & elem){return elem.y;});
    return y;
}

// Return log(binomial_tail(n, k, p))
// binomial_tail(n, k, p) = sum_{i = k}^n bincoef(n, i) * p^i * (1 - p)^{n - i}
// bincoef(n, k) = gamma(n + 1) / (gamma(k + 1) * gamma(n - k + 1))

template <typename I, typename T>
T logbinom(I n, I k, T p)
{
    if (n == k) return n * std::log(p);

    auto term = std::exp(std::lgamma(n + 1) - std::lgamma(k + 1) - std::lgamma(n - k + 1) +
                         k * std::log(p) + (n - k) * std::log(T(1.0) - p));
    auto bin_tail = term;
    auto p_term = p / (T(1.0) - p);

    for (I i = k + 1; i < n + 1; i++)
    {
        term *= (n - i + 1) / i * p_term;
        bin_tail += term;
    }

    return std::log(bin_tail);
}

}

template <typename T>
using pset_t = std::set<std::tuple<Point<int>, T>>;

// Image moments class

template <typename T>
struct Moments
{
    Point<T> pt0;
    T mu, mu_x, mu_y, mu_xx, mu_xy, mu_yy;

    template <typename Pt, typename = std::enable_if_t<std::is_base_of_v<Point<T>,  std::remove_cvref_t<Pt>>>>
    static Moments from_pset(Pt && pt0, const pset_t<T> & pset)
    {
        Moments m {std::forward<Pt>(pt0)};
        for (const auto & [pt, val] : pset)
        {
            m.add_point(pt, val);
        }
        return m;
    }

    static Moments from_pset(const pset_t<T> & pset)
    {
        Point<T> pt0; T mu = T();
        for (const auto & [pt, val] : pset)
        {
            pt0 += val * pt;
            mu += val;
        }

        if (mu) return Moments::from_pset(pt0 / mu, pset);
        else
        {
            Moments m;
            for (const auto & [pt, _] : pset)
            {
                m.pt0 += pt;
            }
            m.pt0 /= pset.size();
            return m;
        }
    }

    Moments operator+(const Moments & m) const
    {
        if (pt0 != m.pt0)
        {
            auto new_m = m.update_seed(pt0);
            return {pt0, mu + new_m.mu, mu_x + new_m.mu_x, mu_y + new_m.mu_y,
                    mu_xx + new_m.mu_xx, mu_xy + new_m.mu_xy, mu_yy + new_m.mu_yy};
        }

        return {pt0, mu + m.mu, mu_x + m.mu_x, mu_y + m.mu_y,
                mu_xx + m.mu_xx, mu_xy + m.mu_xy, mu_yy + m.mu_yy};
    }

    Moments operator-(const Moments & m) const
    {
        if (pt0 != m.pt0)
        {
            auto new_m = m.update_seed(pt0);
            return {pt0, mu - new_m.mu, mu_x - new_m.mu_x, mu_y - new_m.mu_y,
                    mu_xx - new_m.mu_xx, mu_xy - new_m.mu_xy, mu_yy - new_m.mu_yy};
        }

        return {pt0, mu - m.mu, mu_x - m.mu_x, mu_y - m.mu_y,
                mu_xx - m.mu_xx, mu_xy - m.mu_xy, mu_yy - m.mu_yy};
    }

    Moments & operator+=(const Moments & m)
    {
        if (pt0 != m.pt0)
        {
            auto new_m = m.update_seed(pt0);
            mu += new_m.mu; mu_x += new_m.mu_x; mu_y += new_m.mu_y;
            mu_xx += new_m.mu_xx; mu_xy += new_m.mu_xy; mu_yy += new_m.mu_yy;
            return *this;
        }

        mu += m.mu; mu_x += m.mu_x; mu_y += m.mu_y;
        mu_xx += m.mu_xx; mu_xy += m.mu_xy; mu_yy += m.mu_yy;
        return *this;
    }

    Moments & operator-=(const Moments & m)
    {
        if (pt0 != m.pt0)
        {
            auto new_m = m.update_seed(pt0);
            mu -= new_m.mu; mu_x -= new_m.mu_x; mu_y -= new_m.mu_y;
            mu_xx -= new_m.mu_xx; mu_xy -= new_m.mu_xy; mu_yy -= new_m.mu_yy;
            return *this;
        }

        mu -= m.mu; mu_x -= m.mu_x; mu_y -= m.mu_y;
        mu_xx -= m.mu_xx; mu_xy -= m.mu_xy; mu_yy -= m.mu_yy;
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    void add_point(const Point<V> & point, T val)
    {
        auto dist = point - pt0;

        mu += val;
        mu_x += dist.x * val;
        mu_y += dist.y * val;
        mu_xx += dist.x * dist.x * val;
        mu_xy += dist.x * dist.y * val;
        mu_yy += dist.y * dist.y * val;
    }

    Point<T> center_of_mass() const
    {
        if (mu) return {mu_x / mu + pt0.x, mu_y / mu + pt0.y};
        return pt0;
    }

    T theta() const
    {
        if (mu)
        {
            T mx = mu_x / mu;
            T my = mu_y / mu;
            T mxx = mu_xx / mu - mx * mx;
            T mxy = mu_xy / mu - mx * my;
            T myy = mu_yy / mu - my * my;

            T theta = std::atan(2 * mxy / (mxx - myy)) / 2;
            if (myy > mxx) theta += M_PI_2;
            return theta;
        }
        return T();
    }

    Moments update_seed(const Point<T> & pt) const
    {
        auto dist = pt0 - pt;
        return {pt, mu, mu_x + dist.x * mu, mu_y + dist.y * mu,
                mu_xx + 2 * dist.x * mu_x + dist.x * dist.x * mu,
                mu_xy + dist.x * mu_y + dist.y * mu_x + dist.x * dist.y * mu,
                mu_yy + 2 * dist.y * mu_y + dist.y * dist.y * mu};
    }

    friend std::ostream & operator<<(std::ostream & os, const Moments & m)
    {
        os << "{" << m.x0 << ", " << m.y0 << ", " << m.mu << ", " << m.mu_x  << ", " << m.mu_y << ", "
           << m.mu_xx << ", " << m.mu_xy << ", " << m.mu_yy << "}";
        return os;
    }
};

// Line support class

template <typename T>
struct Pixels
{
    Moments<T> moments;
    pset_t<T> pset;

    Pixels() = default;
    Pixels(const pset_t<T> & pset) : moments(Moments<T>::from_pset(pset)), pset(pset) {}
    Pixels(pset_t<T> && pset) : moments(Moments<T>::from_pset(pset)), pset(std::move(pset)) {}

    Pixels merge(const Pixels & pixels) const
    {
        pset_t<T> pint, pmrg;
        std::set_intersection(pset.begin(), pset.end(), pixels.pset.begin(), pixels.pset.end(), std::inserter(pint, pint.begin()));
        std::set_union(pset.begin(), pset.end(), pixels.pset.begin(), pixels.pset.end(), std::inserter(pmrg, pmrg.begin()));

        auto m = moments + pixels.moments - Moments<T>::from_pset(moments.pt0, pint);

        if (m.mu) return {pmrg, m.update_seed(moments.center_of_mass())};
        return {pmrg, m};
    }

    void insert(Pixels && pixels)
    {
        pset_t<T> pint;
        std::set_intersection(pset.begin(), pset.end(), pixels.pset.begin(), pixels.pset.end(), std::inserter(pint, pint.begin()));

        moments += pixels.moments - Moments<T>::from_pset(moments.pt0, pint);

        pset.insert(std::make_move_iterator(pixels.pset.begin()), std::make_move_iterator(pixels.pset.end()));
    }

    Line<T> get_line() const
    {
        if (moments.mu)
        {
            T theta = moments.theta();
            Point<T> tau {std::cos(theta), std::sin(theta)};
            Point<T> ctr = moments.center_of_mass();

            T tmin = std::numeric_limits<T>::max(), tmax = std::numeric_limits<T>::lowest();
            for (const auto & [pt, _] : pset)
            {
                T prod = tau.dot(pt - ctr);
                if (prod < tmin) tmin = prod;
                if (prod > tmax) tmax = prod;
            }

            return {ctr + tmin * tau, ctr + tmax * tau};
        }
        return {moments.pt0, moments.pt0};
    }
};

// Connectivity structure class

struct Structure
{
    int radius, rank;
    std::vector<Point<int>> idxs;

    Structure(int radius, int rank) : radius(radius), rank(rank)
    {
        for (int i = -radius; i <= radius; i++)
        {
            for (int j = -radius; j <= radius; j++)
            {
                if (std::abs(i) + std::abs(j) <= rank) idxs.push_back(Point<int>{j, i});
            }
        }
        std::sort(idxs.begin(), idxs.end());
    }

    std::string info() const
    {
        return "<Structure, radius = " + std::to_string(radius) + ", rank = " + std::to_string(rank) + ">";
    }
};

// Sparse 2D peaks

struct Peaks
{
    std::list<Point<size_t>> points;

    template <typename T>
    Peaks(const array<T> & data, size_t radius, T vmin)
    {
        std::array<size_t, 2> axes {0, 1};
        std::unordered_map<Point<size_t>, Point<size_t>, detail::PointHasher<size_t>> peak_map;
        auto add_peak = [&data, &peak_map, radius, vmin](size_t index)
        {
            auto y = data.index_along_dim(index, 0);
            auto x = data.index_along_dim(index, 1);
            if (data[data.ravel_index({y, x})] > vmin)
            {
                peak_map.try_emplace(Point<size_t>{x / radius, y / radius}, Point<size_t>{x, y});
            }
        };

        for (auto axis : axes)
        {
            for (size_t i = radius / 2; i < data.shape[1 - axis]; i += radius)
            {
                maxima1d(data.line_begin(axis, i), data.line_end(axis, i), add_peak, data, std::array<size_t, 1>{axis});
            }
        }

        std::transform(std::make_move_iterator(peak_map.begin()),
                       std::make_move_iterator(peak_map.end()),
                       std::back_inserter(points),
                       [](std::pair<Point<size_t>, Point<size_t>> && elem){return std::move(elem.second);});

        // Sorting peaks in descending order
        points.sort(
            [&data](Point<size_t> a, Point<size_t> b)
            {
                return data[data.ravel_index({a.y, a.x})] > data[data.ravel_index({b.y, b.x})];
            });
    }

    template <typename T>
    Peaks(py::array_t<T> d, size_t radius, T vmin) : Peaks(array<T>(d.request()), radius, vmin) {}

    template <typename T>
    void filter(const array<T> & data, const Structure & srt, T vmin, size_t npts)
    {
        std::set<Point<size_t>> support;
        std::set<Point<size_t>> new_points;
        auto iter = points.begin();

        while (iter != points.end())
        {
            support.clear(); new_points.clear();

            size_t prev_size = 0;
            support.insert(*iter);

            while (support.size() != prev_size && support.size() < npts)
            {
                prev_size = support.size();

                for (const auto & point: support)
                {
                    for (const auto & shift: srt.idxs)
                    {
                        auto pt = point + shift;

                        if (data.is_inbound({pt.y, pt.x}) && data[data.ravel_index({pt.y, pt.x})] > vmin)
                        {
                            new_points.insert(pt);
                        }
                    }
                }

                support.merge(std::move(new_points)); new_points.clear();
            }

            if (support.size() < npts) iter = points.erase(iter);
            else ++iter;
        }
    }

    void mask(const array<bool> & mask)
    {
        auto iter = points.begin();
        while (iter != points.end())
        {
            if (!mask[mask.ravel_index({iter->y, iter->x})]) iter = points.erase(iter);
            else ++iter;
        }
    }

    std::string info() const
    {
        return "<Peaks, points.shape = (" + std::to_string(points.size()) + ", 2)>";
    }
};

// Final streak class

template <typename T>
struct Streak
{
    Pixels<T> pixels;
    std::vector<Line<T>> linelets;
    Line<T> line;

    Streak(const pset_t<T> & pset) : pixels(pset), line(pixels.get_line())
    {
        linelets.push_back(line);
    }

    Streak(pset_t<T> && pset) : pixels(std::move(pset)), line(pixels.get_line())
    {
        linelets.push_back(line);
    }

    void insert(Streak && streak)
    {
        pixels.insert(std::move(streak.pixels));
        linelets.insert(linelets.end(), std::make_move_iterator(streak.linelets.begin()),
                                        std::make_move_iterator(streak.linelets.end()));
    }

    T log_nfa(unsigned min_size, T xtol, T p) const
    {
        size_t k = 0;
        for (auto ll: linelets)
        {
            if (line.distance(ll.pt0) < xtol && line.distance(ll.pt1) < xtol) k++;
        }
        return -static_cast<int>(min_size) * std::log(p) + detail::logbinom(linelets.size(), k, p);
    }

    void update_line()
    {
        line = pixels.get_line();
    }
};

template <typename T>
struct Pattern
{
    array<T> data;
    array<bool> good;
    Structure structure;

    template <typename Data, typename Good, typename Struct,
        typename = std::enable_if_t<
            std::is_base_of_v<array<T>, std::remove_cvref_t<Data>> &&
            std::is_base_of_v<array<bool>, std::remove_cvref_t<Good>> &&
            std::is_base_of_v<Structure, std::remove_cvref_t<Struct>>
        >
    >
    Pattern(Data && d, Good && g, Struct && s) :
        data(std::forward<Data>(d)), good(std::forward<Good>(g)), structure(std::forward<Struct>(s))
    {
        if (data.ndim != 2)
        {
            throw std::invalid_argument("Pattern data array has invalid number of dimensions (" +
                                        std::to_string(data.ndim) + ")");
        }
        check_equal("data and good have incompatible shapes",
                    data.shape.begin(), data.shape.end(), good.shape.begin(), good.shape.end());
    }

    template <typename Struct,
        typename = std::enable_if_t<
            std::is_base_of_v<Structure, std::remove_cvref_t<Struct>>
        >
    >
    Pattern(const py::array_t<T> & d, const py::array_t<bool> & g, Struct && s) :
        Pattern(array<T>{d.request()}, array<bool>{g.request()}, std::forward<Struct>(s)) {}

    auto find_streaks(Peaks peaks, T xtol, T vmin, T log_eps, unsigned max_iter, unsigned lookahead, unsigned min_size)
    {
        std::vector<std::array<T, 4>> lines;

        while (peaks.points.size())
        {
            Streak<T> streak = get_streak(peaks.points.front(), xtol, vmin, max_iter, lookahead);

            if (streak.log_nfa(min_size, xtol, xtol / (structure.radius + 0.5)) < log_eps)
            {
                update(streak.pixels.pset);
                peaks.mask(good);
                lines.emplace_back(streak.line);
            }
            else peaks.points.pop_front();
        }

        return lines;
    }

    Streak<T> get_streak(int x, int y, T xtol, T vmin, unsigned max_iter, unsigned lookahead) const
    {
        Streak<T> streak {get_pset(x, y)};
        for (unsigned i = 0; i < max_iter; i++)
        {
            auto old_size = streak.linelets.size();

            streak = grow_step(std::move(streak), streak.line.pt0, direction::backward, xtol, vmin, lookahead);
            streak = grow_step(std::move(streak), streak.line.pt1, direction::forward, xtol, vmin, lookahead);

            if (streak.linelets.size() == old_size) break;

            streak.update_line();
        }
        return streak;
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    Streak<T> get_streak(const Point<I> & point, T xtol, T vmin, unsigned max_iter, unsigned lookahead) const
    {
        return get_streak(point.x, point.y, xtol, vmin, max_iter, lookahead);
    }

    pset_t<T> get_pset(int x, int y) const
    {
        pset_t<T> pset;
        for (auto shift : structure.idxs)
        {
            int new_x = x + shift.x, new_y = y + shift.y;
            if (data.is_inbound({new_y, new_x}) && good[good.ravel_index({new_y, new_x})])
            {
                pset.emplace_hint(pset.end(), Point<int>{new_x, new_y}, data[data.ravel_index({new_y, new_x})]);
            }
        }
        return pset;
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    pset_t<T> get_pset(const Point<I> & point) const
    {
        return get_pset(point.x, point.y);
    }

    std::string info() const
    {
        return "<Pattern, data.shape = (" + std::to_string(data.shape[0]) + ", " + std::to_string(data.shape[1]) +
               "), good.shape = (" + std::to_string(good.shape[0]) + ", " + std::to_string(good.shape[1]) +
               "), struct = " + structure.info() + ">";
    }

    void update(const pset_t<T> & pset)
    {
        for (auto [pt, _] : pset)
        {
            good[good.ravel_index({pt.y, pt.x})] = false;
        }
    }

private:
    Point<int> find_next_step(const Streak<T> & streak, const Point<T> & point, int max_cnt, direction dir) const
    {
        BhmIterator<T, int> pix {point.round(), streak.line.norm(), point};
        auto step = bresenham_step(streak.line.tau, dir);

        Point<int> new_step;

        for (int cnt = 0; cnt <= max_cnt; cnt++)
        {
            pix.step(new_step);
            new_step = Point<int>();

            if (pix.is_xnext(step)) new_step.x = step.x;
            if (pix.is_ynext(step)) new_step.y = step.y;
        }

        return pix.point;
    }

    std::pair<bool, Streak<T>> add_streak(Streak<T> && streak, const Point<int> & pt, T xtol, T vmin) const
    {
        Streak new_streak {get_pset(pt.x, pt.y)};
        if (new_streak.line.magnitude())
        {
            T d0 = streak.line.distance(new_streak.line.pt0);
            T d1 = streak.line.distance(new_streak.line.pt1);

            if (d0 < xtol && d1 < xtol)
            {
                Point<int> pt {new_streak.pixels.moments.pt0.round()};
                if (data.is_inbound({pt.y, pt.x}))
                {
                    if (data[data.ravel_index({pt.y, pt.x})] > vmin)
                    {
                        streak.insert(std::move(new_streak));
                        return std::make_pair(true, std::move(streak));
                    }
                }
            }
        }
        return std::make_pair(false, std::move(streak));
    }

    Streak<T> grow_step(Streak<T> && streak, Point<T> point, direction dir, T xtol, T vmin, unsigned lookahead) const
    {
        unsigned max_gap = std::sqrt(streak.line.magnitude()) / structure.radius;
        unsigned tries = 0, gap = 0;

        while (tries < lookahead && gap < max_gap)
        {
            Point<int> pt = find_next_step(streak, point, structure.radius, dir);

            if (!data.is_inbound({pt.y, pt.x})) break;

            if (good[good.ravel_index({pt.y, pt.x})])
            {
                auto [is_add, new_streak] = add_streak(std::move(streak), pt, xtol, vmin);

                if (is_add) return new_streak;
                else
                {
                    streak = std::move(new_streak);
                    tries++;
                }
            }
            else gap++;

            point = pt;
        }

        return streak;
    }
};

template <typename T>
std::vector<std::array<T, 4>> detect_streaks(Peaks peaks, py::array_t<T> data, py::array_t<bool> mask, Structure structure,
                                             T xtol, T vmin, T log_eps, unsigned max_iter, unsigned lookahead, size_t min_size);

}

#endif