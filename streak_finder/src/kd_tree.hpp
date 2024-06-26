#ifndef KD_TREE_
#define KD_TREE_
#include "array.hpp"

namespace streak_finder {

template <typename Point, typename Data>
class KDTree;

template <typename Point, typename Data>
class KDNode
{
public:
    using data_type = Data;
    using item_type = std::pair<Point, Data>;
    using node_ptr = std::shared_ptr<KDNode>;

    item_type item;
    int cut_dim;

    KDNode() = default;

    template <typename Item, typename = std::enable_if_t<std::is_same_v<item_type, std::remove_cvref_t<Item>>>>
    KDNode(Item && item, int dir, node_ptr lt = node_ptr(), node_ptr rt = node_ptr(), node_ptr par = node_ptr()) :
        item(std::forward<Item>(item)), cut_dim(dir), left(lt), right(rt), parent(par) {}

    auto ndim() const -> decltype(std::declval<Point &>().size()) {return item.first.size();}

    Point & point() {return item.first;}
    const Point & point() const {return item.first;}

    Data & data() {return item.second;}
    const Data & data() const {return item.second;}

    template <typename Pt>
    bool is_left(const Pt & pt) const
    {
        return pt[this->cut_dim] < point()[this->cut_dim];
    }

private:
    node_ptr left;
    node_ptr right;
    node_ptr parent;

    friend class KDTree<Point, Data>;
};

template <typename Node, typename Point, typename Data, typename = void>
struct is_node : std::false_type {};

template <typename Node, typename Point, typename Data>
struct is_node<Node, Point, Data,
    typename std::enable_if_t<std::is_base_of_v<KDNode<Point, Data>, std::remove_cvref_t<Node>>>
> : std::true_type {};

template <typename Node, typename F, typename Data>
constexpr bool is_node_v = is_node<Node, F, Data>::value;

template<typename Point, typename Data>
class KDTree
{
public:
    using F = typename Point::value_type;
    using node_t = KDNode<Point, Data>;
    using node_ptr = typename node_t::node_ptr;
    using item_type = typename node_t::item_type;

    class KDIterator
    {
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = KDNode<Point, Data>;
        using difference_type = std::ptrdiff_t;
        using pointer = typename KDNode<Point, Data>::node_ptr;
        using reference = const value_type &;

        KDIterator() : ptr(nullptr), root(nullptr) {}

        bool operator==(const KDIterator & rhs) const
        {
            return root == rhs.root && ptr == rhs.ptr;
        }

        bool operator!=(const KDIterator & rhs) const {return !operator==(rhs);}

        KDIterator & operator++()
        {
            if (!ptr)
            {
                // ++ from end(). Get the root of the tree
                ptr = root;

                // error! ++ requested for an empty tree
                while (ptr && ptr->left) ptr = ptr->left;

            }
            else if (ptr->right)
            {
                // successor is the farthest left node of right subtree
                ptr = ptr->right;

                while (ptr->left) ptr = ptr->left;
            }
            else
            {
                // have already processed the left subtree, and
                // there is no right subtree. move up the tree,
                // looking for a parent for which nodePtr is a left child,
                // stopping if the parent becomes NULL. a non-NULL parent
                // is the successor. if parent is NULL, the original node
                // was the last node inorder, and its successor
                // is the end of the list
                node_ptr p = ptr->parent;
                while (p && ptr == p->right)
                {
                    ptr = p; p = p->parent;
                }

                // if we were previously at the right-most node in
                // the tree, nodePtr = nullptr, and the iterator specifies
                // the end of the list
                ptr = p;
            }

            return *this;
        }

        KDIterator operator++(int)
        {
            auto saved = *this;
            operator++();
            return saved;
        }

        KDIterator & operator--()
        {
            if (!ptr)
            {
                // -- from end(). Get the root of the tree
                ptr = root;

                // move to the largest value in the tree,
                // which is the last node inorder
                while (ptr && ptr->right) ptr = ptr->right;
            }
            else if (ptr->left)
            {
                // must have gotten here by processing all the nodes
                // on the left branch. predecessor is the farthest
                // right node of the left subtree
                ptr = ptr->left;

                while (ptr->right) ptr = ptr->right;
            }
            else
            {
                // must have gotten here by going right and then
                // far left. move up the tree, looking for a parent
                // for which ptr is a right child, stopping if the
                // parent becomes nullptr. a non-nullptr parent is the
                // predecessor. if parent is nullptr, the original node
                // was the first node inorder, and its predecessor
                // is the end of the list
                node_ptr p = ptr->parent;
                while (p && ptr == p->left)
                {
                    ptr = p; p = p->parent;
                }

                // if we were previously at the left-most node in
                // the tree, ptr = NULL, and the iterator specifies
                // the end of the list
                ptr = p;
            }

            return *this;
        }

        KDIterator operator--(int)
        {
            auto saved = *this;
            operator--();
            return saved;
        }

        reference operator*() const {return *ptr;}
        pointer operator->() const {return ptr;}

    private:
        friend class KDTree<Point, Data>;

        node_ptr ptr;
        node_ptr root;

        KDIterator(node_ptr ptr, node_ptr root) : ptr(ptr), root(root) {}
    };

    class Rectangle
    {
    public:
        std::vector<F> low, high;

        Rectangle() = default;

        Rectangle(const KDTree<Point, Data> & tree)
        {
            for (size_t i = 0; i < tree.ndim(); i++)
            {
                low.push_back(tree.find_min(i)->point()[i]);
                high.push_back(tree.find_max(i)->point()[i]);
            }
        }

        void update(const Point & pt)
        {
            for (size_t i = 0; i < pt.size(); i++)
            {
                low[i] = std::min(low[i], pt[i]);
                high[i] = std::max(high[i], pt[i]);
            }
        }

        template <typename Pt, typename G = std::common_type_t<F, typename Pt::value_type>>
        G distance(const Pt & pt) const
        {
            G dist = G();
            for (size_t i = 0; i < pt.size(); i++)
            {
                if (pt[i] < low[i]) dist += std::pow(low[i] - pt[i], 2);
                if (pt[i] > high[i]) dist += std::pow(pt[i] - high[i], 2);
            }
            return dist;
        }

    private:
        friend class KDTree<Point, Data>;

        Rectangle trim_left(node_ptr node) const
        {
            Rectangle rect = *this;
            rect.high[node->cut_dim] = node->point()[node->cut_dim];
            return rect;
        }

        Rectangle trim_right(node_ptr node) const
        {
            Rectangle rect = *this;
            rect.low[node->cut_dim] = node->point()[node->cut_dim];
            return rect;
        }
    };

    using const_iterator = KDIterator;
    using iterator = const_iterator;

    using rect_t = Rectangle;
    using rect_ptr = std::shared_ptr<rect_t>;

    template <typename T>
    using query_t = std::pair<const_iterator, T>;
    template <typename T>
    using stack_t = std::vector<std::pair<const_iterator, T>>;

    KDTree() : root(nullptr), rect(nullptr) {}

    KDTree(std::vector<item_type> && items) : KDTree()
    {
        root = build_tree(std::make_move_iterator(items.begin()),
                          std::make_move_iterator(items.end()), node_ptr(), 0);
        if (root) rect = std::make_shared<rect_t>(*this);
    }

    auto ndim() const -> decltype(std::declval<Point &>().size())
    {
        if (!root) return 0;
        else return root->ndim();
    }

    bool empty() const {return !root;}

    void clear()
    {
        root = node_ptr();
        rect = rect_ptr();
    }

    const_iterator begin() const
    {
        return {begin_node(root), root};
    }

    const_iterator end() const
    {
        return {node_ptr(), root};
    }

    const_iterator insert(item_type && item)
    {
        const_iterator inserted;
        if (root) std::tie(root, inserted) = insert_node(root, std::move(item), root, root->cut_dim);
        else std::tie(root, inserted) = insert_node(root, std::move(item), root, 0);

        if (inserted != end())
        {
            if (!rect) rect = std::make_shared<rect_t>(*this);
            else rect->update(item.first);
        }

        return inserted;
    }

    size_t erase(const Point & pt)
    {
        size_t removed;
        std::tie(root, removed) = remove_node(root, pt);

        if (rect && removed)
        {
            if (root)
            {
                for (size_t i = 0; i < ndim(); i++)
                {
                    if (pt[i] == rect->low[i]) rect->low[i] = find_min(i)->point()[i];
                    if (pt[i] == rect->high[i]) rect->high[i] = find_max(i)->point()[i];
                }
            }
            else rect = rect_ptr();
        }

        return removed;
    }

    const_iterator erase(const_iterator pos)
    {
        if (pos != end())
        {
            erase((pos++)->point());
        }
        return pos;
    }

    const_iterator find(const Point & pt) const
    {
        return {find_node(root, pt, *rect, node_ptr()), root};
    }

    const_iterator find_min(int axis) const
    {
        return {find_min_node(root, axis), root};
    }

    const_iterator find_max(int axis) const
    {
        return {find_max_node(root, axis), root};
    }

    template <typename Pt, typename G = std::common_type_t<F, typename Pt::value_type>>
    query_t<G> find_nearest(const Pt & pt) const
    {
        return nearest_node(root, pt, *rect, {const_iterator(root, root), std::numeric_limits<G>::max()});
    }

    template <typename Pt, typename G = std::common_type_t<F, typename Pt::value_type>>
    stack_t<G> find_range(const Pt & pt, G range_sq) const
    {
        return find_range_node(root, pt, range_sq, *rect, {});
    }

    void print() const
    {
        print_node(std::cout, root);
        print_rect(std::cout);
    }

private:
    node_ptr root;
    rect_ptr rect;

    template <class Iter>
    node_ptr build_tree(Iter first, Iter last, node_ptr par, int dir)
    {
        using value_t = typename std::iterator_traits<Iter>::value_type;

        if (last <= first) return node_ptr();
        else if (last == std::next(first))
        {
            return std::make_shared<node_t>(*first, dir, node_ptr(), node_ptr(), par);
        }
        else
        {
            auto compare = [dir](const value_t & a, const value_t & b){return a.first[dir] < b.first[dir];};
            auto iter = wirthmedian(first, last, compare);

            node_ptr node = std::make_shared<node_t>(*iter, dir, node_ptr(), node_ptr(), par);
            node->left = build_tree(first, iter, node, (dir + 1) % node->ndim());
            node->right = build_tree(std::next(iter), last, node, (dir + 1) % node->ndim());
            return node;
        }
    }

    std::tuple<node_ptr, const_iterator> insert_node(node_ptr node, item_type && item, node_ptr par, int dir)
    {
        // Create new node if empty
        if (!node)
        {
            node = std::make_shared<node_t>(std::move(item), dir, node_ptr(), node_ptr(), par);
            return {node, const_iterator(node, root)};
        }

        // Duplicate data point, no insertion
        if (item.first == node->point())
        {
            return {node, end()};
        }

        const_iterator inserted;

        if (node->is_left(item.first))
        {
            // left of splitting line
            std::tie(node->left, inserted) = insert_node(node->left, std::move(item), node, (node->cut_dim + 1) % node->ndim());
        }
        else
        {
            // on or right of splitting line
            std::tie(node->right, inserted) = insert_node(node->right, std::move(item), node, (node->cut_dim + 1) % node->ndim());
        }

        return {node, inserted};
    }

    std::tuple<node_ptr, size_t> remove_node(node_ptr node, const Point & pt)
    {
        // Fell out of tree
        if (!node) return {node, 0};

        size_t removed;

        // Found the node
        if (node->point() == pt)
        {
            // Take replacement from right
            if (node->right)
            {
                // Swapping the node
                node->item = find_min_node(node->right, node->cut_dim)->item;

                std::tie(node->right, removed) = remove_node(node->right, node->point());
            }
            // Take replacement from left
            else if (node->left)
            {
                // Swapping the nodes
                node->item = find_min_node(node->left, node->cut_dim)->item;

                // move left subtree to right!
                std::tie(node->right, removed) = remove_node(node->left, node->point());
                // left subtree is now empty
                node->left = node_ptr();
            }
            // Remove this leaf
            else
            {
                node = node_ptr(); removed = 1;
            }
        }
        // Search left subtree
        else if (node->is_left(pt))
        {
            std::tie(node->left, removed) = remove_node(node->left, pt);
        }
        // Search right subtree
        else std::tie(node->right, removed) = remove_node(node->right, pt);

        return {node, removed};
    }

    node_ptr min_node(node_ptr a, node_ptr b, node_ptr c, int axis) const
    {
        if (b && b->point()[axis] < a->point()[axis])
        {
            if (c && c->point()[axis] < b->point()[axis]) return c;
            return b;
        }
        if (c && c->point()[axis] < a->point()[axis]) return c;
        return a;
    }

    node_ptr max_node(node_ptr a, node_ptr b, node_ptr c, int axis) const
    {
        if (b && b->point()[axis] > a->point()[axis])
        {
            if (c && c->point()[axis] > b->point()[axis]) return c;
            return b;
        }
        if (c && c->point()[axis] > a->point()[axis]) return c;
        return a;
    }

    node_ptr find_min_node(node_ptr node, int axis) const
    {
        // Fell out of tree
        if (!node) return node;

        if (node->cut_dim == axis)
        {
            if (!node->left) return node;
            else return find_min_node(node->left, axis);
        }
        else return min_node(node, find_min_node(node->left, axis), find_min_node(node->right, axis), axis);
    }

    node_ptr find_max_node(node_ptr node, int axis) const
    {
        // Fell out of tree
        if (!node) return node;

        if (node->cut_dim == axis)
        {
            if (!node->right) return node;
            else return find_max_node(node->right, axis);
        }
        else return max_node(node, find_max_node(node->left, axis), find_max_node(node->right, axis), axis);
    }

    node_ptr begin_node(node_ptr node) const
    {
        if (!node) return node;
        if (!node->left) return node;
        return begin_node(node->left);
    }

    template <typename Point1, typename Point2, typename G = std::common_type_t<typename Point1::value_type, typename Point2::value_type>>
    G distance(const Point1 & a, const Point2 & b) const
    {
        G dist = G();
        for (size_t i = 0; i < a.size(); i++) dist += std::pow(a[i] - b[i], 2);
        return dist;
    }

    node_ptr find_node(node_ptr node, const Point & pt, const rect_t & rect, node_ptr query) const
    {
        // Fell out of tree
        if (!node) return query;
        // This cell is too far away
        if (rect.distance(pt) > F()) return query;

        // We found the node
        if (pt == node->point()) query = node;

        // pt is close to left child
        if (node->is_left(pt))
        {
            // First left then right
            query = find_node(node->left, pt, rect.trim_left(node), query);
            query = find_node(node->right, pt, rect.trim_right(node), query);
        }
        // pt is closer to right child
        else
        {
            // First right then left
            query = find_node(node->right, pt, rect.trim_right(node), query);
            query = find_node(node->left, pt, rect.trim_left(node), query);
        }

        return query;
    }

    template <typename Pt, typename G = std::common_type_t<F, typename Pt::value_type>>
    query_t<G> nearest_node(node_ptr node, const Pt & pt, const rect_t & rect, query_t<G> && query) const
    {
        // Fell out of tree
        if (!node) return query;
        // This cell is too far away
        if (rect.distance(pt) >= query.second) return query;

        // Update if the root is closer
        auto dist_sq = distance(node->point(), pt);
        if (dist_sq < query.second) query = std::make_pair(const_iterator(node, root), dist_sq);

        // pt is close to left child
        if (node->is_left(pt))
        {
            // First left then right
            query = nearest_node(node->left, pt, rect.trim_left(node), std::move(query));
            query = nearest_node(node->right, pt, rect.trim_right(node), std::move(query));
        }
        // pt is closer to right child
        else
        {
            // First right then left
            query = nearest_node(node->right, pt, rect.trim_right(node), std::move(query));
            query = nearest_node(node->left, pt, rect.trim_left(node), std::move(query));
        }

        return query;
    }

    template <typename Pt, typename G = std::common_type_t<F, typename Pt::value_type>>
    stack_t<G> stack_push_node(node_ptr node, const Pt & pt, stack_t<G> && stack) const
    {
        if (node->left) stack = stack_push_node(node->left, pt, std::move(stack));
        stack.emplace_back(const_iterator(node, root), distance(node->point(), pt));
        if (node->right) stack = stack_push_node(node->right, pt, std::move(stack));
        return stack;
    }

    template <typename Pt, typename G = std::common_type_t<F, typename Pt::value_type>>
    stack_t<G> find_range_node(node_ptr node, const Pt & pt, G range_sq, const rect_t & rect, stack_t<G> && stack) const
    {
        // Fell out of tree
        if (!node) return stack;
        // The cell doesn't overlap the query
        if (rect.distance(pt) > range_sq) return stack;

        // The query contains the cell
        if (distance(pt, rect.low) < range_sq && distance(pt, rect.high) < range_sq)
        {
            return stack_push_node(node, pt, std::move(stack));
        }

        auto dist_sq = distance(pt, node->point());
        // Put this item to stack
        if (dist_sq < range_sq) stack.emplace_back(const_iterator(node, root), dist_sq);

        // Search left subtree
        stack = find_range_node(node->left, pt, range_sq, rect.trim_left(node), std::move(stack));

        // Search right subtree
        stack = find_range_node(node->right, pt, range_sq, rect.trim_right(node), std::move(stack));

        return stack;
    }

    std::ostream & print_rect(std::ostream & os) const
    {
        os << "low  : [";
        std::copy(rect->low.begin(), rect->low.end(), std::experimental::make_ostream_joiner(os, ", "));
        os << "]" << std::endl;

        os << "high : [";
        std::copy(rect->high.begin(), rect->high.end(), std::experimental::make_ostream_joiner(os, ", "));
        os << "]" << std::endl;
        return os;
    }

    std::ostream & print_node(std::ostream & os, node_ptr node, size_t level = 0) const
    {
        if (!node) return os;

        print_node(os, node->left, level + 1);

        os << std::string(level, '\t') << "(";
        std::copy(node->point().begin(), node->point().end(), std::experimental::make_ostream_joiner(os, ", "));
        os << ")" << " axis = " << node->cut_dim << std::endl;

        print_node(os, node->right, level + 1);
        return os;
    }
};

}

#endif
