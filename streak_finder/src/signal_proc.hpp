#ifndef SIGNAL_PROC_
#define SIGNAL_PROC_
#include "array.hpp"

namespace streak_finder {

template <typename InputIt, typename T, typename Axes, class UnaryFunction>
UnaryFunction maxima_nd(InputIt first, InputIt last, UnaryFunction unary_op, const array<T> & arr, const Axes & axes, size_t order)
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

            while (ahead != last && *ahead == *iter) ++ahead;

            if (*ahead < *iter)
            {
                auto index = arr.index(iter);

                size_t n = 1;
                for (; n < axes.size(); n++)
                {
                    auto coord = arr.index_along_dim(index, axes[n]);
                    if (coord > 1 && coord < arr.shape[axes[n]] - 1)
                    {
                        if (arr[index - arr.stride(axes[n])] < *iter && arr[index + arr.stride(axes[n])] < *iter)
                        {
                            continue;
                        }
                    }

                    break;
                }

                if (n >= order) unary_op(index);

                // Skip samples that can't be maximum, check if it's not last
                if (ahead != last) iter = ahead;
            }
        }

        iter = std::next(iter);
    }

    return unary_op;
}

}

#endif
