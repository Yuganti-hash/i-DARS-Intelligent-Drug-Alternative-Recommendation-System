"""
b_tree.py — B-Tree Implementation for Disease Database Indexing

A B-Tree of order `t` (minimum degree) ensures:
  - Every node has at most 2t-1 keys
  - Every non-root node has at least t-1 keys
  - All leaves are at the same depth
  - Search: O(log n)
  - Insert: O(log n)

Each key in the B-Tree is a disease name (string).
Each key maps to a list of MedicineNode objects (via AVL tree stored in BTreeNode).
"""

from avl_tree import AVLTree


class BTreeNode:
    """
    A single node in the B-Tree.
    
    Attributes:
        keys        : sorted list of disease name strings (the index keys)
        avl_trees   : parallel list of AVLTree objects — avl_trees[i] holds all
                      medicines associated with keys[i]
        children    : list of child BTreeNode pointers (len = len(keys)+1 for internal nodes)
        is_leaf     : True if this node has no children
        n           : current number of keys stored in the node
    """

    def __init__(self, is_leaf=True):
        self.keys = []          # disease name strings
        self.avl_trees = []     # AVLTree per disease (parallel to keys)
        self.children = []      # child BTreeNode pointers
        self.is_leaf = is_leaf
        self.n = 0              # number of active keys


class BTree:
    """
    B-Tree of minimum degree `t`.
    
    - Minimum keys per non-root node : t - 1
    - Maximum keys per node          : 2t - 1
    - Minimum children per non-leaf  : t
    - Maximum children per non-leaf  : 2t
    """

    def __init__(self, t=3):
        """
        Args:
            t (int): Minimum degree. Higher t → fewer levels, better for disk I/O.
                     t=3 means nodes hold 2–5 keys; good balance for in-memory use.
        """
        self.t = t
        self.root = BTreeNode(is_leaf=True)

    # ------------------------------------------------------------------ #
    # PUBLIC API                                                           #
    # ------------------------------------------------------------------ #

    def insert_disease(self, disease_name: str, medicine=None):
        """
        Insert a disease into the B-Tree (or add a medicine to an existing disease).
        
        If the root is full (2t-1 keys), split it first (increases tree height by 1).
        Time complexity: O(t · log_t n)
        """
        root = self.root

        # Case 1: disease already exists → just add medicine to its AVL tree
        existing_avl = self._search_node(root, disease_name)
        if existing_avl is not None:
            if medicine is not None:
                existing_avl.insert(medicine)
            return

        # Case 2: root is full → split before inserting
        if root.n == 2 * self.t - 1:
            new_root = BTreeNode(is_leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0, self.root)
            self.root = new_root

        self._insert_non_full(self.root, disease_name, medicine)

    def search(self, disease_name: str):
        """
        Search for a disease and return its AVLTree (medicine store).
        Returns None if not found.
        Time complexity: O(t · log_t n)
        """
        return self._search_node(self.root, disease_name)

    def get_all_diseases(self):
        """
        In-order traversal to collect all disease names.
        Time complexity: O(n)
        """
        result = []
        self._inorder(self.root, result)
        return result

    # ------------------------------------------------------------------ #
    # PRIVATE HELPERS                                                      #
    # ------------------------------------------------------------------ #

    def _search_node(self, node: BTreeNode, key: str):
        """
        Recursively search for `key` starting at `node`.
        Returns the AVLTree for that key, or None.
        """
        i = 0
        # Find the first key >= key
        while i < node.n and key > node.keys[i]:
            i += 1

        if i < node.n and key == node.keys[i]:
            return node.avl_trees[i]   # found!

        if node.is_leaf:
            return None                # not found, no children

        # Descend into the appropriate child
        return self._search_node(node.children[i], key)

    def _insert_non_full(self, node: BTreeNode, key: str, medicine):
        """
        Insert `key` into a node guaranteed to be non-full.
        Splits children on the way down so we never need to backtrack.
        """
        i = node.n - 1

        if node.is_leaf:
            # Shift keys right to make room, then insert in sorted position
            node.keys.append(None)
            node.avl_trees.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.avl_trees[i + 1] = node.avl_trees[i]
                i -= 1
            node.keys[i + 1] = key
            new_avl = AVLTree()
            if medicine is not None:
                new_avl.insert(medicine)
            node.avl_trees[i + 1] = new_avl
            node.n += 1
        else:
            # Find the child to descend into
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1  # child index

            # If that child is full, split it first
            if node.children[i].n == 2 * self.t - 1:
                self._split_child(node, i, node.children[i])
                # After split, the median moved up; decide which sub-child to take
                if key > node.keys[i]:
                    i += 1

            self._insert_non_full(node.children[i], key, medicine)

    def _split_child(self, parent: BTreeNode, i: int, child: BTreeNode):
        """
        Split child at index `i` of `parent`.
        
        The child has 2t-1 keys. After split:
          - Left  child keeps keys[0 .. t-2]  (t-1 keys)
          - Median key[t-1] is promoted to parent at position i
          - Right child gets keys[t .. 2t-2] (t-1 keys)
        
        Time complexity: O(t)
        """
        t = self.t
        new_node = BTreeNode(is_leaf=child.is_leaf)

        # Median key goes up to parent
        median_key = child.keys[t - 1]
        median_avl = child.avl_trees[t - 1]

        # Right child gets the upper half of child's keys
        new_node.keys = child.keys[t:]
        new_node.avl_trees = child.avl_trees[t:]
        new_node.n = t - 1

        if not child.is_leaf:
            new_node.children = child.children[t:]
            child.children = child.children[:t]

        # Left child (original) keeps the lower half
        child.keys = child.keys[:t - 1]
        child.avl_trees = child.avl_trees[:t - 1]
        child.n = t - 1

        # Insert median into parent at position i
        parent.keys.insert(i, median_key)
        parent.avl_trees.insert(i, median_avl)
        parent.children.insert(i + 1, new_node)
        parent.n += 1

    def _inorder(self, node: BTreeNode, result: list):
        """
        In-order traversal of B-Tree collecting all disease names.
        For a B-Tree: visit child[0], key[0], child[1], key[1], ..., child[n]
        """
        for i in range(node.n):
            if not node.is_leaf:
                self._inorder(node.children[i], result)
            result.append(node.keys[i])
        if not node.is_leaf:
            self._inorder(node.children[node.n], result)
