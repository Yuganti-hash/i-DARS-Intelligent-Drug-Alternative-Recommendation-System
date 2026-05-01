"""
avl_tree.py — AVL Tree Implementation for Medicine Storage

An AVL Tree is a self-balancing Binary Search Tree where the heights of the
two child subtrees of any node differ by at most 1 (balance factor ∈ {-1, 0, 1}).

After every insert, rotations restore this invariant:
  - Left Rotation (LL case)
  - Right Rotation (RR case)
  - Left-Right Rotation (LR case)
  - Right-Left Rotation (RL case)

Medicines are keyed by their `name` (string) for BST ordering.

Time complexities:
  - Insert : O(log n)
  - Search : O(log n)
  - Traversal : O(n)
"""


class AVLNode:
    """
    A single node in the AVL Tree representing one medicine.
    
    Attributes:
        medicine : a MedicineNode (dict-like) object with all drug attributes
        key      : medicine name string (BST ordering key)
        left     : left child AVLNode
        right    : right child AVLNode
        height   : height of the subtree rooted here (leaf = 1)
    """

    def __init__(self, medicine):
        self.medicine = medicine
        self.key = medicine.name          # BST key = medicine name
        self.left = None
        self.right = None
        self.height = 1                   # new node is a leaf


class AVLTree:
    """
    AVL Tree storing MedicineNode objects, keyed by medicine name.
    
    Each BTreeNode (disease) owns one AVLTree that organises all
    associated medicines in O(log n) insert / search order.
    """

    def __init__(self):
        self.root = None

    # ------------------------------------------------------------------ #
    # PUBLIC API                                                           #
    # ------------------------------------------------------------------ #

    def insert(self, medicine):
        """
        Insert a medicine into the AVL tree.
        Duplicate names overwrite the existing node.
        Time complexity: O(log n)
        """
        self.root = self._insert(self.root, medicine)

    def search(self, name: str):
        """
        Search for a medicine by name.
        Returns the MedicineNode or None.
        Time complexity: O(log n)
        """
        node = self._search(self.root, name)
        return node.medicine if node else None

    def get_all_medicines(self):
        """
        In-order traversal: returns all medicines sorted by name.
        Time complexity: O(n)
        """
        result = []
        self._inorder(self.root, result)
        return result

    def filter_by_age_group(self, age_group: str):
        """
        Traverse the entire AVL tree and return medicines suitable
        for the given age_group ("Child" | "Adult" | "Senior").
        Time complexity: O(n)
        """
        result = []
        self._filter_inorder(self.root, age_group, result)
        return result

    # ------------------------------------------------------------------ #
    # CORE PRIVATE METHODS                                                 #
    # ------------------------------------------------------------------ #

    def _get_height(self, node):
        """Return height of a node (0 for None)."""
        return node.height if node else 0

    def _get_balance(self, node):
        """
        Balance Factor = height(left) - height(right)
        BF ∈ {-1, 0, 1} → balanced
        BF > 1           → left-heavy  (needs right rotation or LR rotation)
        BF < -1          → right-heavy (needs left rotation or RL rotation)
        """
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _update_height(self, node):
        """Recompute height from children heights."""
        node.height = 1 + max(self._get_height(node.left),
                              self._get_height(node.right))

    # ------------------------------------------------------------------ #
    # ROTATIONS                                                            #
    # ------------------------------------------------------------------ #

    def _rotate_right(self, z):
        """
        Right Rotation (LL imbalance — left subtree is left-heavy):
        
              z                y
             / \             /   \
            y   T4   →      x     z
           / \             / \   / \
          x   T3          T1 T2 T3  T4
         / \
        T1  T2
        
        Time complexity: O(1)
        """
        y = z.left
        T3 = y.right

        # Perform rotation
        y.right = z
        z.left = T3

        # Update heights (z is now lower, update first)
        self._update_height(z)
        self._update_height(y)

        return y  # new root of this subtree

    def _rotate_left(self, z):
        """
        Left Rotation (RR imbalance — right subtree is right-heavy):
        
          z                  y
         / \               /   \
        T1   y     →      z     x
            / \          / \   / \
           T2   x       T1 T2 T3  T4
               / \
              T3  T4
        
        Time complexity: O(1)
        """
        y = z.right
        T2 = y.left

        # Perform rotation
        y.left = z
        z.right = T2

        # Update heights
        self._update_height(z)
        self._update_height(y)

        return y  # new root of this subtree

    # ------------------------------------------------------------------ #
    # INSERT                                                               #
    # ------------------------------------------------------------------ #

    def _insert(self, node, medicine):
        """
        Standard BST insert followed by AVL rebalancing.
        Returns the (possibly new) root of this subtree.
        """
        # --- Step 1: Standard BST insertion ---
        if node is None:
            return AVLNode(medicine)

        if medicine.name < node.key:
            node.left = self._insert(node.left, medicine)
        elif medicine.name > node.key:
            node.right = self._insert(node.right, medicine)
        else:
            # Duplicate key → overwrite medicine data
            node.medicine = medicine
            return node

        # --- Step 2: Update height ---
        self._update_height(node)

        # --- Step 3: Get balance factor ---
        balance = self._get_balance(node)

        # --- Step 4: Rebalance if needed (4 cases) ---

        # Case A: Left-Left (LL) → single right rotation
        if balance > 1 and medicine.name < node.left.key:
            return self._rotate_right(node)

        # Case B: Right-Right (RR) → single left rotation
        if balance < -1 and medicine.name > node.right.key:
            return self._rotate_left(node)

        # Case C: Left-Right (LR) → left rotate left child, then right rotate node
        if balance > 1 and medicine.name > node.left.key:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)

        # Case D: Right-Left (RL) → right rotate right child, then left rotate node
        if balance < -1 and medicine.name < node.right.key:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        # Tree is already balanced at this node
        return node

    # ------------------------------------------------------------------ #
    # SEARCH & TRAVERSAL                                                   #
    # ------------------------------------------------------------------ #

    def _search(self, node, name: str):
        """Standard BST search by medicine name."""
        if node is None or node.key == name:
            return node
        if name < node.key:
            return self._search(node.left, name)
        return self._search(node.right, name)

    def _inorder(self, node, result: list):
        """In-order (sorted) traversal collecting medicine objects."""
        if node is None:
            return
        self._inorder(node.left, result)
        result.append(node.medicine)
        self._inorder(node.right, result)

    def _filter_inorder(self, node, age_group: str, result: list):
        """
        In-order traversal that only appends medicines whose
        `suitable_for` list contains the requested age_group.
        """
        if node is None:
            return
        self._filter_inorder(node.left, age_group, result)
        if age_group in node.medicine.suitable_for:
            result.append(node.medicine)
        self._filter_inorder(node.right, age_group, result)
