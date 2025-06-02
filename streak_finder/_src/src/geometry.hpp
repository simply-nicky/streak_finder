#ifndef GEOMETRY_
#define GEOMETRY_
#include "array.hpp"

namespace streak_finder {

template <typename T, size_t N>
struct PointND : public std::array<T, N>
{
    static_assert(std::is_arithmetic_v<T>);

    template <typename V, typename = std::enable_if_t<std::is_constructible_v<T, V>>>
    operator PointND<V, N>() const
    {
        PointND<V, N> res;
        for (size_t i = 0; i < N; i++) res[i] = static_cast<V>(this->operator[](i));
        return res;
    }

    // In-place operators

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    PointND & operator+=(const PointND<V, N> & rhs) &
    {
        for (auto & x: *this) x += rhs[std::addressof(x) - this->data()];
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    PointND & operator+=(V rhs) &
    {
        for (auto & x: *this) x += rhs;
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    PointND & operator-=(const PointND<V, N> & rhs) &
    {
        for (auto & x: *this) x -= rhs[std::addressof(x) - this->data()];
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    PointND & operator-=(V rhs) &
    {
        for (auto & x: *this) x -= rhs;
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    PointND & operator*=(const PointND<V, N> & rhs) &
    {
        for (auto & x: *this) x *= rhs[std::addressof(x) - this->data()];
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    PointND & operator*=(V rhs) &
    {
        for (auto & x: *this) x *= rhs;
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    PointND & operator/=(const PointND<V, N> & rhs) &
    {
        for (auto & x: *this) x /= rhs[std::addressof(x) - this->data()];
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    PointND & operator/=(V rhs) &
    {
        for (auto & x: *this) x /= rhs;
        return *this;
    }

    // friend operators

    template <typename V, typename U = std::common_type_t<T, V>>
    friend PointND<U, N> operator+(const PointND & lhs, const PointND<V, N> & rhs)
    {
        PointND<U, N> result = lhs;
        result += rhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    friend PointND<U, N> operator+(const PointND & lhs, V rhs)
    {
        PointND<U, N> result = lhs;
        result += rhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    friend PointND<U, N> operator+(V lhs, const PointND & rhs)
    {
        PointND<U, N> result = rhs;
        result += lhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    friend PointND<U, N> operator-(const PointND & lhs, const PointND<V, N> & rhs)
    {
        PointND<U, N> result = lhs;
        result -= rhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    friend PointND<U, N> operator-(const PointND & lhs, V rhs)
    {
        PointND<U, N> result = lhs;
        result -= rhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    friend PointND<U, N> operator-(V lhs, const PointND & rhs)
    {
        PointND<U, N> result = rhs;
        result -= lhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    friend PointND<U, N> operator*(const PointND & lhs, const PointND<V, N> & rhs)
    {
        PointND<U, N> result = lhs;
        result *= rhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    friend PointND<U, N> operator*(const PointND & lhs, V rhs)
    {
        PointND<U, N> result = lhs;
        result *= rhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    friend PointND<U, N> operator*(V lhs, const PointND & rhs)
    {
        PointND<U, N> result = rhs;
        result *= lhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    friend PointND<U, N> operator/(const PointND & lhs, const PointND<V, N> & rhs)
    {
        PointND<U, N> result = lhs;
        result /= rhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    friend PointND<U, N> operator/(const PointND & lhs, V rhs)
    {
        PointND<U, N> result = lhs;
        result /= rhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    friend PointND<U, N> operator/(V lhs, const PointND & rhs)
    {
        PointND<U, N> result = rhs;
        result /= lhs;
        return result;
    }

    friend std::ostream & operator<<(std::ostream & os, const PointND & pt)
    {
        os << "{";
        std::copy(pt.begin(), pt.end(), std::experimental::make_ostream_joiner(os, ", "));
        os << "}";
        return os;
    }

    // methods

    PointND<T, N> clamp(const PointND<T, N> & lo, const PointND<T, N> & hi) const
    {
        PointND<T, N> result;
        for (size_t i = 0; i < N; i++) result[i] = std::clamp(this->operator[](i), lo[i], hi[i]);
        return result;
    }

    std::array<T, N> coordinate() const
    {
        std::array<T, N> result = to_array();
        std::reverse(result.begin(), result.end());
        return result;
    }

    PointND<T, N> round() const
    {
        auto result = *this;
        for (auto & x : result) x = std::round(x);
        return result;
    }

    std::array<T, N> & to_array() & {return *this;}
    const std::array<T, N> & to_array() const & {return *this;}
    std::array<T, N> && to_array() && {return std::move(*this);}

    T & x() requires(N >= 1) {return this->operator[](0);}
    T & y() requires(N >= 2) {return this->operator[](1);}

    const T & x() const requires(N >= 1) {return this->operator[](0);}
    const T & y() const requires(N >= 2) {return this->operator[](1);}
};

template <template <typename, size_t> class Array, typename T, size_t ... sizes>
auto concatenate(const Array<T, sizes> & ... arrays)
{
    Array<T, (sizes + ...)> result;
    size_t index {};

    ((std::copy_n(arrays.begin(), sizes, result.begin() + index), index += sizes), ...);

    return result;
}

template <typename T, typename V, size_t N, typename U = std::common_type_t<T, V>>
U distance(const PointND<T, N> & a, const PointND<V, N> & b)
{
    U dist = U();
    for (size_t i = 0; i < N; i++) dist += (a[i] - b[i]) * (a[i] - b[i]);
    return dist;
}

template <template <typename, size_t> class Array, typename T, typename V, typename U = std::common_type_t<T, V>, size_t N>
U dot(const Array<T, N> & a, const Array<V, N> & b)
{
    U res = U();
    for (size_t i = 0; i < N; i++) res += a[i] * b[i];
    return res;
}

template <template <typename, size_t> class Array, typename T, size_t N>
T magnitude(const Array<T, N> & a)
{
    T res = T();
    for (size_t i = 0; i < N; i++) res += a[i] * a[i];
    return res;
}

template <template <typename, size_t> class Array, typename T, size_t N>
auto amplitude(const Array<T, N> & a) -> decltype(std::sqrt(std::declval<T &>()))
{
    return std::sqrt(magnitude(a));
}

template <size_t N, class Container, typename T = typename Container::value_type>
constexpr PointND<T, N> to_point(Container & a, size_t start)
{
    return apply_to_sequence<N>([&a, start](auto... idxs) -> PointND<T, N> {return {a[start + idxs]...};});
}

template <typename T>
using Point = PointND<T, 2>;

template <typename T, size_t N>
struct LineND
{
    PointND<T, N> pt0, pt1;

    LineND() = default;

    template <typename Pt0, typename Pt1, typename = std::enable_if_t<
        std::is_base_of_v<PointND<T, N>, std::remove_cvref_t<Pt0>> &&
        std::is_base_of_v<PointND<T, N>, std::remove_cvref_t<Pt1>>
    >>
    LineND(Pt0 && pt0, Pt1 && pt1) : pt0(std::forward<Pt0>(pt0)), pt1(std::forward<Pt1>(pt1)) {}

    bool operator==(const LineND<T, N> & rhs) const {return pt0 == rhs.pt0 && pt1 == rhs.pt1;}
    bool operator!=(const LineND<T, N> & rhs) const {return !operator==(rhs);}

    PointND<T, N> normal() const requires(N >= 2)
    {
        PointND<T, N> norm {};
        norm[0] = pt1[1] - pt0[1];
        norm[1] = pt0[0] - pt1[0];
        return norm;
    }

    PointND<T, N> tangent() const {return pt1 - pt0;}

    template <typename V, typename U = std::common_type_t<T, V>, typename W = decltype(std::sqrt(std::declval<V &>()))>
    PointND<W, N> project_to_streak(const PointND<V, N> & point) const
    {
        auto tau = tangent();
        auto mag = magnitude(tau);

        if (mag)
        {
            auto center = 0.5 * (pt0 + pt1);
            auto r = point - center;
            auto r_tau = static_cast<W>(dot(tau, r)) / mag;
            return std::clamp<W>(r_tau, -0.5, 0.5) * tau + center;
        }
        return pt0;
    }

    template <typename V, typename U = std::common_type_t<T, V>, typename W = decltype(std::sqrt(std::declval<V &>()))>
    W distance(const PointND<V, N> & point) const
    {
        return amplitude(point - project_to_streak(point));
    }

    template <typename V, typename U = std::common_type_t<T, V>, typename W = decltype(std::sqrt(std::declval<V &>()))>
    PointND<W, N> project_to_line(const PointND<V, N> & point) const
    {
        auto tau = tangent();
        auto mag = magnitude(tau);

        if (mag)
        {
            auto center = 0.5 * (pt0 + pt1);
            auto r = point - center;
            auto r_tau = static_cast<W>(dot(tau, r)) / mag;
            return r_tau * tau + center;
        }
        return pt0;
    }

    template <typename V, typename U = std::common_type_t<T, V>, typename W = decltype(std::sqrt(std::declval<V &>()))>
    W normal_distance(const PointND<V, N> & point) const
    {
        return amplitude(point - project_to_line(point));
    }

    friend std::ostream & operator<<(std::ostream & os, const LineND<T, N> & line)
    {
        os << "{" << line.pt0 << ", " << line.pt1 << "}";
        return os;
    }

    std::array<T, 2 * N> to_array() const {return concatenate(pt0.to_array(), pt1.to_array());}
    std::pair<PointND<T, N>, PointND<T, N>> to_pair() const & {return std::make_pair(pt0, pt1);}
    std::pair<PointND<T, N>, PointND<T, N>> to_pair() && {return std::make_pair(std::move(pt0), std::move(pt1));}
};

template <typename T>
using Line = LineND<T, 2>;

namespace detail{

template <typename T, size_t N = 2>
struct PointHasher
{
    size_t operator()(const PointND<T, N> & point) const
    {
        size_t h = 0;
        for (size_t i = 0; i < N; i++) h = detail::hash_combine(h, point[i]);
        return h;
    }
};

}

template <size_t N, typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
std::array<I, N + 1> normalise_shape(const std::vector<I> & shape)
{
    if (shape.size() < N)
        fail_container_check("wrong number of dimensions (" + std::to_string(shape.size()) +
                             " < " + std::to_string(N) + ")", shape);
    std::array<I, N + 1> res {std::reduce(shape.begin(), std::prev(shape.end(), N), I(1), std::multiplies())};
    for (size_t i = 0; i < N; i++) res[i + 1] = shape[shape.size() - N + i];
    return res;
}

template <size_t N>
class UniquePairs
{
public:
    constexpr static size_t NumPairs = N * (N - 1) / 2;

    static const UniquePairs & instance()
    {
        static UniquePairs axes;
        return axes;
    }

    UniquePairs(const UniquePairs &)        = delete;
    void operator=(const UniquePairs &)     = delete;

    const std::pair<size_t, size_t> & pairs(size_t index) const {return m_pairs[index];}
    const std::array<size_t, N - 1> & indices(size_t axis) const {return m_lookup[axis];}

private:
    std::array<std::pair<size_t, size_t>, NumPairs> m_pairs;
    std::array<std::array<size_t, N - 1>, N> m_lookup;

    UniquePairs()
    {
        std::pair<size_t, size_t> pair {};
        for (size_t i = 0; i < NumPairs; i++)
        {
            ++pair.second;
            if (pair.second == N)
            {
                ++pair.first;
                pair.second = pair.first + 1;
            }
            m_pairs[i] = pair;
        }

        for (size_t i = 0; i < N; i++)
        {
            size_t index = 0;
            for (size_t j = 0; j < NumPairs; j++)
            {
                if (m_pairs[j].first == i || m_pairs[j].second == i) m_lookup[i][index++] = j;
            }
        }
    }
};

}

#endif
