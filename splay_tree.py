"""
splay_tree.py — Splay Tree (MRU Cache) for Recently Queried Diseases

A Splay Tree is a self-adjusting BST where every accessed node is moved
("splayed") to the root via a series of rotations. This gives:

  - Amortised O(log n) per operation
  - O(1) access for recently accessed nodes (working-set property)
  - No extra balance information needed per node

This makes it ideal as a Most-Recently-Used (MRU) cache:
  - Querying "Seasonal Flu" repeatedly → stays at root → near O(1) lookup.

Operations implemented:
  - insert(key)      : O(log n) amortised
  - access(key)      : O(log n) amortised; splays found node to root
  - get_mru()        : O(1) — just return root
  - get_top_k(k)     : O(k log n) — in-order traversal limited to k nodes
"""


class SplayNode:
    """
    Node in a Splay Tree representing a cached disease query.
    
    Attributes:
        key         : disease name string
        access_count: how many times this disease has been queried
        left, right : child pointers
    """

    def __init__(self, key: str):
        self.key = key
        self.access_count = 1
        self.left = None
        self.right = None


class SplayTree:
    """
    Splay Tree used as an MRU (Most Recently Used) cache.
    
    Every time a disease is queried via the recommendation engine,
    call `access(disease_name)`. The splay operation bubbles it to
    the root so subsequent accesses are faster.
    """

    def __init__(self):
        self.root = None

    # ------------------------------------------------------------------ #
    # PUBLIC API                                                           #
    # ------------------------------------------------------------------ #

    def access(self, key: str):
        """
        Access (or insert) a disease key.
        Splays the node to the root.
        Increments access_count.
        Time complexity: O(log n) amortised
        """
        if self.root is None:
            self.root = SplayNode(key)
            return

        self.root = self._splay(self.root, key)

        if self.root.key == key:
            # Node was found and is now root
            self.root.access_count += 1
        else:
            # Key not found — insert it as new root
            new_node = SplayNode(key)
            if key < self.root.key:
                new_node.right = self.root
                new_node.left = self.root.left
                self.root.left = None
            else:
                new_node.left = self.root
                new_node.right = self.root.right
                self.root.right = None
            self.root = new_node

    def get_mru(self):
        """Return the most recently used disease name (root key). O(1)."""
        return self.root.key if self.root else None

    def get_top_k(self, k: int):
        """
        Return the top-k most accessed diseases.
        Collects all nodes, sorts by access_count descending.
        Time complexity: O(n log n) due to sort.
        """
        all_nodes = []
        self._collect(self.root, all_nodes)
        all_nodes.sort(key=lambda n: n.access_count, reverse=True)
        return [(n.key, n.access_count) for n in all_nodes[:k]]

    # ------------------------------------------------------------------ #
    # SPLAY OPERATION                                                      #
    # ------------------------------------------------------------------ #

    def _splay(self, root, key: str):
        """
        Splay `key` to root using top-down splay.
        
        Three cases per zig-step:
          1. Zig       : key is in root's left/right child → single rotation
          2. Zig-Zig   : key and root.child are both left (or both right)
                         → rotate parent first, then rotate root
          3. Zig-Zag   : key is left of root but right of parent (or vice-versa)
                         → two opposite rotations
        
        We use the "top-down" approach with two auxiliary trees (left_tree, right_tree)
        to avoid recursive stack overhead.
        
        Time complexity: O(log n) amortised, O(n) worst case on a single op.
        """
        if root is None:
            return None

        # Header node acts as a dummy parent to assemble left/right subtrees
        header = SplayNode("")
        left_tree = header    # max of the left part
        right_tree = header   # min of the right part
        t = root

        while True:
            if key < t.key:
                if t.left is None:
                    break
                # Zig-Zig left: key < t.key and key < t.left.key
                if key < t.left.key:
                    t = self._rotate_right(t)   # rotate grandparent up
                    if t.left is None:
                        break
                # Link right: attach t to right_tree (t is > key)
                right_tree.left = t
                right_tree = t
                t = t.left
                right_tree.left = None

            elif key > t.key:
                if t.right is None:
                    break
                # Zig-Zig right: key > t.key and key > t.right.key
                if key > t.right.key:
                    t = self._rotate_left(t)    # rotate grandparent up
                    if t.right is None:
                        break
                # Link left: attach t to left_tree
                left_tree.right = t
                left_tree = t
                t = t.right
                left_tree.right = None

            else:
                break  # found!

        # Assemble
        left_tree.right = t.left
        right_tree.left = t.right
        t.left = header.right
        t.right = header.left

        return t

    # ------------------------------------------------------------------ #
    # ROTATIONS                                                            #
    # ------------------------------------------------------------------ #

    def _rotate_right(self, node: SplayNode) -> SplayNode:
        """
        Right rotation:
              y              x
             / \            / \
            x   C   →      A   y
           / \                / \
          A   B              B   C
        Time complexity: O(1)
        """
        x = node.left
        node.left = x.right
        x.right = node
        return x

    def _rotate_left(self, node: SplayNode) -> SplayNode:
        """
        Left rotation:
            x                  y
           / \               /   \
          A   y     →       x     C
             / \           / \
            B   C         A   B
        Time complexity: O(1)
        """
        y = node.right
        node.right = y.left
        y.left = node
        return y

    # ------------------------------------------------------------------ #
    # TRAVERSAL                                                            #
    # ------------------------------------------------------------------ #

    def _collect(self, node: SplayNode, result: list):
        """Pre-order traversal to collect all nodes."""
        if node is None:
            return
        result.append(node)
        self._collect(node.left, result)
        self._collect(node.right, result)
