"""
fibonacci_heap.py — Fibonacci Heap for Priority-Based Drug Ranking

A Fibonacci Heap supports the following amortised time complexities:
  - insert        : O(1)
  - find_min      : O(1)
  - extract_min   : O(log n)  amortised
  - decrease_key  : O(1)      amortised
  - merge (union) : O(1)

Since we need MAX scores (highest effectiveness/match wins), we negate
all scores before insertion to simulate a max-heap via a min-heap:
  stored_key = -actual_score

So `extract_min` on stored_key ≡ `extract_max` on actual_score.

Structure:
  - Root list: a doubly-linked circular list of heap-ordered trees
  - Each node has: key, value (medicine), degree, mark, parent, child,
    left, right pointers.
"""

import math


class FibNode:
    """
    Node in the Fibonacci Heap.
    
    Attributes:
        key     : float — negated score (for max-heap simulation)
        value   : the medicine object or (medicine, score) tuple
        degree  : number of children
        mark    : True if this node has lost a child since becoming a child itself
        parent  : parent FibNode (None if in root list)
        child   : one of the children (doubly linked circular list)
        left    : left sibling in doubly linked circular list
        right   : right sibling in doubly linked circular list
    """

    def __init__(self, key: float, value):
        self.key = key
        self.value = value
        self.degree = 0
        self.mark = False
        self.parent = None
        self.child = None
        # Self-loop for circular doubly linked list
        self.left = self
        self.right = self


class FibonacciHeap:
    """
    Fibonacci Heap implementation.
    
    Internally a min-heap on stored keys (= negated scores).
    Public interface uses `insert_max` and `extract_max` for clarity.
    """

    def __init__(self):
        self.min_node = None    # pointer to current minimum (= highest score)
        self.n = 0              # total number of nodes

    # ------------------------------------------------------------------ #
    # PUBLIC MAX-HEAP API                                                  #
    # ------------------------------------------------------------------ #

    def insert_max(self, score: float, medicine):
        """
        Insert a medicine with its recommendation score.
        We negate the score so the min-heap gives us the maximum-score item.
        Time complexity: O(1) amortised
        """
        self._insert(-score, medicine)

    def extract_max(self):
        """
        Extract the medicine with the highest recommendation score.
        Returns (medicine, score) or None if heap is empty.
        Time complexity: O(log n) amortised
        """
        node = self._extract_min()
        if node is None:
            return None
        return node.value, -node.key    # un-negate the score

    def decrease_key_max(self, node: FibNode, new_higher_score: float):
        """
        Increase the actual score of a node (= decrease its negated stored key).
        Used when a medicine's score is revised upward.
        Time complexity: O(1) amortised
        """
        self._decrease_key(node, -new_higher_score)

    def is_empty(self) -> bool:
        return self.min_node is None

    def size(self) -> int:
        return self.n

    def peek_max(self):
        """Return (medicine, score) of best candidate without removing. O(1)."""
        if self.min_node is None:
            return None
        return self.min_node.value, -self.min_node.key

    # ------------------------------------------------------------------ #
    # CORE OPERATIONS                                                      #
    # ------------------------------------------------------------------ #

    def _insert(self, key: float, value):
        """
        Insert a new node into the root list.
        
        Steps:
          1. Create a new FibNode with given key/value
          2. Add it to the root list (O(1) doubly-linked list insert)
          3. Update min pointer if needed
        
        Time complexity: O(1)
        """
        node = FibNode(key, value)
        self._add_to_root_list(node)
        if self.min_node is None or node.key < self.min_node.key:
            self.min_node = node
        self.n += 1
        return node  # return so caller can use decrease_key later

    def _extract_min(self):
        """
        Extract and return the minimum node (highest actual score).
        
        Steps:
          1. Add all children of min_node to the root list
          2. Remove min_node from root list
          3. Consolidate: merge trees of the same degree
             (uses auxiliary array indexed by degree)
          4. Find new minimum in consolidated root list
        
        Time complexity: O(log n) amortised
        """
        z = self.min_node
        if z is None:
            return None

        # Step 1: Promote all children of z to root list
        if z.child is not None:
            children = self._get_siblings(z.child)
            for child in children:
                self._add_to_root_list(child)
                child.parent = None

        # Step 2: Remove z from root list
        self._remove_from_root_list(z)
        self.n -= 1

        if z == z.right:
            # z was the only node in root list
            self.min_node = None
        else:
            self.min_node = z.right   # temporary; consolidate will fix this
            self._consolidate()

        return z

    def _consolidate(self):
        """
        Consolidate the root list so no two trees have the same degree.
        
        Uses an array A where A[d] holds the unique tree of degree d.
        When two trees have the same degree, link the one with larger key
        under the one with smaller key (heap-order property).
        
        The maximum degree after consolidation is at most floor(log_φ(n))
        where φ = (1+√5)/2 ≈ 1.618.
        
        Time complexity: O(log n) amortised (D(n) ≈ log_φ n consolidations)
        """
        # Maximum possible degree: log_φ(n) + 2 for safety
        max_degree = int(math.log(self.n + 1) / math.log(1.618)) + 2 if self.n > 0 else 1
        A = [None] * (max_degree + 1)

        # Collect all roots first (snapshot, since we'll modify list)
        roots = self._get_siblings(self.min_node)

        for w in roots:
            x = w
            d = x.degree

            # Grow array if needed (safety for unusual cases)
            while d >= len(A):
                A.append(None)

            while A[d] is not None:
                y = A[d]   # another tree with same degree
                if x.key > y.key:
                    x, y = y, x   # x should be the one with smaller key
                # Make y a child of x (link operation)
                self._heap_link(y, x)
                A[d] = None
                d += 1

                # Extend A if degree grew beyond current size
                while d >= len(A):
                    A.append(None)

            A[d] = x

        # Rebuild root list and find new min
        self.min_node = None
        for node in A:
            if node is not None:
                # Re-add to root list (re-initialise sibling pointers)
                node.left = node
                node.right = node
                if self.min_node is None:
                    self.min_node = node
                else:
                    self._add_to_root_list(node)
                    if node.key < self.min_node.key:
                        self.min_node = node

    def _heap_link(self, y: FibNode, x: FibNode):
        """
        Make y a child of x.
        
        Precondition: x.key <= y.key (heap order)
        Steps:
          1. Remove y from root list
          2. Add y to x's child list
          3. Increment x.degree
          4. Clear y.mark (y is now a fresh child)
        
        Time complexity: O(1)
        """
        self._remove_from_root_list(y)
        y.parent = x

        if x.child is None:
            x.child = y
            y.left = y
            y.right = y
        else:
            # Insert y into x's child circular list
            self._add_to_child_list(x, y)

        x.degree += 1
        y.mark = False

    def _decrease_key(self, x: FibNode, new_key: float):
        """
        Decrease the key of node x to new_key.
        
        Steps:
          1. Set x.key = new_key
          2. If heap order violated (x.key < parent.key):
             a. Cut x from parent → add to root list
             b. Cascading cut: if parent was already marked, cut it too
                (and repeat up the tree)
             This limits total cuts to O(1) amortised.
        
        Time complexity: O(1) amortised
        """
        if new_key > x.key:
            raise ValueError("New key is greater than current key")

        x.key = new_key
        y = x.parent

        if y is not None and x.key < y.key:
            self._cut(x, y)
            self._cascading_cut(y)

        if x.key < self.min_node.key:
            self.min_node = x

    def _cut(self, x: FibNode, y: FibNode):
        """
        Remove x from y's child list and add x to root list.
        Time complexity: O(1)
        """
        # Remove x from y's child list
        if x.right == x:
            y.child = None
        else:
            if y.child == x:
                y.child = x.right
            x.left.right = x.right
            x.right.left = x.left

        y.degree -= 1
        # Reset x's sibling pointers and add to root
        x.left = x
        x.right = x
        x.parent = None
        x.mark = False
        self._add_to_root_list(x)

    def _cascading_cut(self, y: FibNode):
        """
        Propagate cuts upward.
        
        If y is unmarked: mark it (first child loss).
        If y is marked  : cut y too, then cascading-cut y's parent.
        
        This ensures each node loses at most 1 child before being cut,
        keeping the tree degree bounded at O(log n).
        
        Time complexity: O(1) amortised
        """
        z = y.parent
        if z is not None:
            if not y.mark:
                y.mark = True
            else:
                self._cut(y, z)
                self._cascading_cut(z)

    # ------------------------------------------------------------------ #
    # DOUBLY-LINKED LIST HELPERS                                           #
    # ------------------------------------------------------------------ #

    def _add_to_root_list(self, node: FibNode):
        """
        Insert `node` into the root circular doubly-linked list.
        Inserts to the right of min_node (or makes it a singleton).
        Time complexity: O(1)
        """
        if self.min_node is None:
            node.left = node
            node.right = node
        else:
            node.right = self.min_node.right
            node.left = self.min_node
            self.min_node.right.left = node
            self.min_node.right = node

    def _remove_from_root_list(self, node: FibNode):
        """
        Remove `node` from the root circular doubly-linked list.
        Time complexity: O(1)
        """
        node.left.right = node.right
        node.right.left = node.left

    def _add_to_child_list(self, parent: FibNode, node: FibNode):
        """
        Insert `node` into parent's child circular list.
        Time complexity: O(1)
        """
        node.right = parent.child.right
        node.left = parent.child
        parent.child.right.left = node
        parent.child.right = node

    def _get_siblings(self, start: FibNode) -> list:
        """
        Collect all nodes in the circular list starting at `start`.
        Returns a plain Python list (snapshot before modification).
        Time complexity: O(k) where k is list length.
        """
        nodes = []
        current = start
        while True:
            nodes.append(current)
            current = current.right
            if current == start:
                break
        return nodes
