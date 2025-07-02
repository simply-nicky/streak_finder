#ifndef SIGNAL_PROC_
#define SIGNAL_PROC_
#include "array.hpp"

namespace streak_finder {

/*----------------------------------------------------------------------------*/
/*------------------------- Bilinear interpolation ---------------------------*/
/*----------------------------------------------------------------------------*/

/* points (integer) follow the convention:      [..., k, j, i], where {i <-> x, j <-> y, k <-> z}
   coordinates (float) follow the convention:   [x, y, z, ...]
 */

template <typename T, typename Coord, typename U = typename Coord::value_type>
T bilinear(const array<T> & arr, const std::vector<array<U>> & grid, const Coord & coord)
{
    std::vector<size_t> lbound, ubound;
    std::vector<T> dx;

    for (size_t n = 0; n < coord.size(); n++)
    {
        auto index = coord.size() - 1 - n;
        // liter is GREATER OR EQUAL
        auto liter = std::lower_bound(grid[index].begin(), grid[index].end(), coord[index]);
        // uiter is GREATER
        auto uiter = std::upper_bound(grid[index].begin(), grid[index].end(), coord[index]);
        // lbound is LESS OR EQUAL
        lbound.push_back(std::clamp<size_t>(std::distance(grid[index].begin(), uiter) - 1, 0, grid[index].size() - 1));
        // rbound is GREATER OR EQUAL
        ubound.push_back(std::clamp<size_t>(std::distance(grid[index].begin(), liter), 0, grid[index].size() - 1));
    }

    for (size_t n = 0; n < coord.size(); n++)
    {
        auto index = coord.size() - 1 - n;
        if (lbound[index] != ubound[index])
        {
            dx.push_back((coord[n] - grid[n][lbound[index]]) / (grid[n][ubound[index]] - grid[n][lbound[index]]));
        }
        else dx.push_back(T());
    }

    T out = T();
    std::vector<size_t> point (coord.size());

    // Iterating over a square around coord
    for (size_t i = 0; i < (1ul << coord.size()); i++)
    {
        T factor = 1.0;
        for (size_t n = 0; n < coord.size(); n++)
        {
            // If the index is odd
            if ((i >> n) & 1)
            {
                point[point.size() - 1 - n] = ubound[ubound.size() - 1 - n];
                factor *= dx[n];
            }
            else
            {
                point[point.size() - 1 - n] = lbound[lbound.size() - 1 - n];
                factor *= 1.0 - dx[n];
            }

        }

        if (arr.is_inbound(point)) out += factor * arr.at(point);
        else
        {
            std::ostringstream oss;
            oss << "Invalid index: {";
            std::copy(point.begin(), point.end(), std::experimental::make_ostream_joiner(oss, ", "));
            oss << "}";
            throw std::runtime_error(oss.str());
        }
    }

    return out;
}

template <typename InputIt, typename T, typename Axes, class UnaryFunction, typename = std::enable_if_t<
    (std::is_base_of_v<typename array<T>::iterator, InputIt> || std::is_base_of_v<typename array<T>::const_iterator, InputIt>) &&
    std::is_invocable_v<std::remove_cvref_t<UnaryFunction>, size_t> && std::is_integral_v<typename Axes::value_type>
>>
UnaryFunction maxima_nd(InputIt first, InputIt last, UnaryFunction && unary_op, const array<T> & arr, const Axes & axes, size_t order)
{
    // First element can't be a maximum
    auto iter = (first != last) ? std::next(first) : first;
    last = (iter != last) ? std::prev(last) : last;

    while (iter != last)
    {
        if (*std::prev(iter) < *iter)
        {
            // ahead can be last
            auto ahead = std::next(iter);

            if (*ahead < *iter)
            {
                // It will return an index relative to the arr.begin() since it's stride is smaller
                auto index = std::addressof(*iter) - arr.data();

                size_t n = 1;
                for (; n < axes.size(); n++)
                {
                    auto coord = arr.index_along_dim(index, axes[n]);
                    if (coord > 1 && coord < arr.shape(axes[n]) - 1)
                    {
                        if (arr[index - arr.strides(axes[n])] < *iter && arr[index + arr.strides(axes[n])] < *iter)
                        {
                            continue;
                        }
                    }

                    break;
                }

                if (n >= order) std::forward<UnaryFunction>(unary_op)(index);

                // Skip samples that can't be maximum, check if it's not last
                if (ahead != last) iter = ahead;
            }
        }

        iter = std::next(iter);
    }

    return std::forward<UnaryFunction>(unary_op);
}

}

#endif
