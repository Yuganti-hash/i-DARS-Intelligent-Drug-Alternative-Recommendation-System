# Complete Conceptual & Line-by-Line Explanation
# Advanced Drug Alternative Recommendation System

---

## TABLE OF CONTENTS

1. [Project Architecture Overview](#1-project-architecture-overview)
2. [Core Computer Science Concepts Used](#2-core-computer-science-concepts-used)
3. [b_tree.py — Line by Line](#3-b_treepy--line-by-line)
4. [avl_tree.py — Line by Line](#4-avl_treepy--line-by-line)
5. [splay_tree.py — Line by Line](#5-splay_treepy--line-by-line)
6. [fibonacci_heap.py — Line by Line](#6-fibonacci_heappy--line-by-line)
7. [recommendation_engine.py — Line by Line](#7-recommendation_enginepy--line-by-line)
8. [generate_dataset.py — Line by Line](#8-generate_datasetpy--line-by-line)
9. [ui_app.py — Line by Line](#9-ui_apppy--line-by-line)
10. [How All Files Connect Together](#10-how-all-files-connect-together)
11. [Time Complexity Summary](#11-time-complexity-summary)

---

## 1. PROJECT ARCHITECTURE OVERVIEW

This system answers one question: **"Given these symptoms / this medicine, what should a patient consider as an alternative?"**

To answer it, four data structures work together in a pipeline:

```
CSV File (2200 rows)
       │
       ▼
  RecommendationEngine.load_csv()
       │
       ├──► B-Tree ──────────────────────────────────── Disease Index
       │         └── each disease key holds an AVL Tree
       │                   └── each AVL node holds a MedicineNode
       │
       ├──► Splay Tree ──────────────────────────────── MRU Cache
       │         (which disease was searched most recently/frequently)
       │
       └──► Fibonacci Heap ──────────────────────────── Priority Ranker
                 (ranks candidate medicines by computed score)
```

**Path A query flow:**
User gives: disease + baseline medicine + age group
→ B-Tree finds the disease → AVL Tree filters by age → FibHeap ranks alternatives

**Path B query flow:**
User gives: symptom text + age group
→ NLP Mapper converts text to disease → same B-Tree/AVL/FibHeap pipeline

---

## 2. CORE COMPUTER SCIENCE CONCEPTS USED

### 2.1 Binary Search Tree (BST)
A BST is a tree where:
- Every node has at most 2 children (left, right)
- All values in the LEFT subtree are SMALLER than the node's value
- All values in the RIGHT subtree are GREATER than the node's value

This ordering means you can search in O(log n) — you never need to check
every element, just follow the "smaller/larger" rule at each node.

**Used in:** AVL Tree (which IS a BST) and Splay Tree (which IS a BST)

### 2.2 Tree Height and Balance Factor
The HEIGHT of a node = the length of the longest path from that node to a leaf.
- Leaf nodes have height = 1
- An empty subtree has height = 0

The BALANCE FACTOR of a node = height(left child) - height(right child)
- BF = 0: perfectly balanced at this node
- BF = 1 or -1: acceptably balanced
- BF = 2 or -2: UNBALANCED → must rotate to fix

**Used in:** AVL Tree to decide when and which rotation to apply

### 2.3 Tree Rotations
A rotation is a local restructuring of 3 nodes that:
- Maintains the BST ordering property (left < root < right)
- Changes the height distribution to restore balance
- Takes O(1) time — only pointer updates, no data copying

There are 4 rotation patterns:
- **LL (Left-Left):** new node inserted far left → single right rotation
- **RR (Right-Right):** new node inserted far right → single left rotation
- **LR (Left-Right):** new node at left-right zigzag → left rotate left child, then right rotate
- **RL (Right-Left):** new node at right-left zigzag → right rotate right child, then left rotate

**Used in:** AVL Tree (balance maintenance) and Splay Tree (splaying)

### 2.4 Amortised Analysis
Amortised analysis averages the cost of operations over a long sequence.
A single splay operation might take O(n) in the worst case, BUT any sequence
of m operations on a splay tree takes O(m log n) TOTAL — so the AMORTISED
cost per operation is O(log n).

Think of it like this: an expensive operation always "pays back" by leaving
the tree in a shape that makes future operations cheaper.

**Used in:** Splay Tree and Fibonacci Heap complexity analysis

### 2.5 Doubly Linked Circular List
A doubly linked list where:
- Every node has a `left` and `right` pointer
- The last node's `right` points back to the first node (circular)
- The first node's `left` points to the last node (circular)

This allows O(1) insertion and removal anywhere, and O(1) traversal
of the full list without knowing its length in advance.

**Used in:** Fibonacci Heap root list and child lists

### 2.6 Potential Function (Fibonacci Heap)
The Fibonacci Heap's amortised analysis uses a potential function Φ:
  Φ = (number of trees in root list) + 2 × (number of marked nodes)

When an operation is cheap (insert → O(1) actual), the potential increases.
When an operation is expensive (extract_min → does consolidation), it
uses that stored potential. This is why the amortised cost stays O(log n).

### 2.7 Greedy Algorithm
The scoring functions in the recommendation engine use a greedy weighted formula:
compute a score for each candidate independently and pick the top-N.
There is no global optimisation or dynamic programming — each candidate is
scored in isolation then ranked.

### 2.8 Multi-Layer NLP (Natural Language Processing)
The symptom mapper uses rule-based NLP — no machine learning model.
It applies 7 transformation layers in sequence (spell fix → synonym expand →
suffix strip → jargon translate → phrase match → regex patterns → token overlap)
and uses a voting system: each layer that fires adds points to a disease candidate.
The disease with the highest total score wins.

---

## 3. b_tree.py — LINE BY LINE

```python
from avl_tree import AVLTree
```
**Line 1:** Import the AVLTree class. This is needed because each key (disease)
in the B-Tree stores not just a string but an entire AVL Tree of medicines.

---

### BTreeNode class

```python
class BTreeNode:
```
A class representing one node in the B-Tree. In a B-Tree, a SINGLE node
can hold MULTIPLE keys (unlike a BST which holds exactly one key per node).
This makes B-Trees very efficient for disk-based storage and minimises
the number of levels needed to store n items.

```python
    def __init__(self, is_leaf=True):
```
Constructor. `is_leaf=True` means when a new node is created, it's assumed
to have no children. Internal nodes will have `is_leaf=False`.

```python
        self.keys = []
```
A Python list storing disease name strings IN SORTED ORDER.
Example: `["Arthritis", "Diabetes", "Migraine"]`
Keeping them sorted is what makes binary search within the node possible.

```python
        self.avl_trees = []
```
A list of AVLTree objects PARALLEL to `self.keys`.
`avl_trees[i]` is the medicine store for `keys[i]`.
So if `keys[0] = "Arthritis"`, then `avl_trees[0]` is the AVL tree
containing all medicines for Arthritis.

```python
        self.children = []
```
List of child BTreeNode pointers. For an internal node (not a leaf),
there is always exactly ONE MORE child than there are keys.
So if a node has 3 keys, it has 4 children.
The rule: all keys in `children[i]` are BETWEEN `keys[i-1]` and `keys[i]`.

```python
        self.is_leaf = is_leaf
```
Boolean flag. If True, `self.children` is empty (this node has no subtrees).

```python
        self.n = 0
```
Counter of how many keys are currently stored in this node.
This is used instead of `len(self.keys)` for efficiency.

---

### BTree class

```python
class BTree:
```
The B-Tree container class. Manages the root and all insert/search operations.

```python
    def __init__(self, t=3):
```
`t` is the MINIMUM DEGREE of the B-Tree. It controls the branching factor:
- Each non-root node holds between t-1 and 2t-1 keys
- Each non-leaf node has between t and 2t children
- With t=3: nodes hold 2 to 5 keys and have 3 to 6 children

Why t=3? It's a good balance for in-memory use — wide enough to keep the
tree shallow, not so wide that linear search within a node is slow.

```python
        self.t = t
        self.root = BTreeNode(is_leaf=True)
```
The tree starts with a single empty leaf node as the root.

---

### insert_disease method

```python
    def insert_disease(self, disease_name: str, medicine=None):
```
Public method to add a disease (and optionally its first medicine).

```python
        root = self.root
```
Local variable for the current root — we may replace it if a split occurs.

```python
        existing_avl = self._search_node(root, disease_name)
        if existing_avl is not None:
            if medicine is not None:
                existing_avl.insert(medicine)
            return
```
First check if the disease already exists. If it does, just add the
medicine to its existing AVL tree. No B-Tree restructuring needed.
This is the fast path — O(log n) search, then O(log k) AVL insert.

```python
        if root.n == 2 * self.t - 1:
```
Check if the root is FULL (has maximum 2t-1 keys, which is 5 for t=3).
When the root is full, we CANNOT insert into it directly — we must split it first.

```python
            new_root = BTreeNode(is_leaf=False)
```
Create a new root node. It will be an internal node (not a leaf).

```python
            new_root.children.append(self.root)
```
The old root becomes the FIRST CHILD of the new root.

```python
            self._split_child(new_root, 0, self.root)
```
Split the old root (which is now child[0] of new_root).
This promotes the MEDIAN KEY of the old root up to new_root.
After this, new_root has exactly 1 key and 2 children.

```python
            self.root = new_root
```
The new root is now the actual root of the B-Tree.
This is the ONLY way the height of a B-Tree increases — always from the top.

```python
        self._insert_non_full(self.root, disease_name, medicine)
```
Now that we've guaranteed the root is not full, do the actual insert.

---

### _search_node method

```python
    def _search_node(self, node: BTreeNode, key: str):
```
Recursive search. Returns the AVLTree for the disease, or None.

```python
        i = 0
        while i < node.n and key > node.keys[i]:
            i += 1
```
Linear scan through keys in THIS NODE to find the first key >= `key`.
Example: keys = ["Arthritis", "Diabetes", "Migraine"], key = "Diabetes"
- i=0: "Diabetes" > "Arthritis" → i=1
- i=1: "Diabetes" > "Diabetes" is FALSE → stop
- i=1 now points to "Diabetes"

Note: This is O(t) linear search within one node. For large t, binary
search could be used, but t=3 makes this negligible.

```python
        if i < node.n and key == node.keys[i]:
            return node.avl_trees[i]
```
Exact match found — return the AVL tree for this disease.

```python
        if node.is_leaf:
            return None
```
We're at a leaf with no match — disease doesn't exist in the tree.

```python
        return self._search_node(node.children[i], key)
```
Not a leaf and no match → descend into child[i].
Why child[i]? Because all keys in child[i] are between keys[i-1] and keys[i],
which is exactly where our key would belong.

---

### _insert_non_full method

```python
    def _insert_non_full(self, node: BTreeNode, key: str, medicine):
```
Insert into a node that is GUARANTEED to have room (n < 2t-1).
Splits children proactively on the way DOWN so we never need to backtrack.

```python
        i = node.n - 1
```
Start from the rightmost existing key and work leftward.

```python
        if node.is_leaf:
```
**Leaf case:** This is where new keys actually get placed in a B-Tree.

```python
            node.keys.append(None)
            node.avl_trees.append(None)
```
Extend both parallel lists by one slot.

```python
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.avl_trees[i + 1] = node.avl_trees[i]
                i -= 1
```
Shift all keys GREATER THAN `key` one position to the right.
This makes room for the new key in sorted position.
Example: keys=["Arthritis","Migraine"], inserting "Diabetes":
- i=1: "Diabetes" < "Migraine" → shift "Migraine" right → keys=["Arthritis","Migraine","Migraine"]
- i=0: "Diabetes" < "Arthritis" is FALSE → stop
- i=-1 (actually i=0 after loop)

```python
            node.keys[i + 1] = key
            new_avl = AVLTree()
            if medicine is not None:
                new_avl.insert(medicine)
            node.avl_trees[i + 1] = new_avl
            node.n += 1
```
Place the key and its new AVL tree at position i+1. Increment count.

```python
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
```
**Internal node case:** Find which child to descend into.
After this loop, child[i] is where `key` belongs.

```python
            if node.children[i].n == 2 * self.t - 1:
                self._split_child(node, i, node.children[i])
                if key > node.keys[i]:
                    i += 1
```
If that child is FULL, split it BEFORE descending.
After the split, the median moved up to position i in the parent.
We check if our key is larger than that median; if so, we go to the right
half (child[i+1]) instead of the left half (child[i]).

```python
            self._insert_non_full(node.children[i], key, medicine)
```
Recurse into the chosen child.

---

### _split_child method

```python
    def _split_child(self, parent: BTreeNode, i: int, child: BTreeNode):
```
Split a FULL child into two halves and promote the median to parent.

`parent` = the node whose child we're splitting
`i` = index of the child in parent.children
`child` = the full node being split

```python
        t = self.t
        new_node = BTreeNode(is_leaf=child.is_leaf)
```
The new node will be a sibling of `child` (it goes to the right of `child`).
It's a leaf if and only if `child` was a leaf.

```python
        median_key = child.keys[t - 1]
        median_avl = child.avl_trees[t - 1]
```
The MEDIAN is the middle key (index t-1 for a 2t-1 key node).
With t=3 and 5 keys (indices 0,1,2,3,4): median is at index 2.
This median will be PROMOTED to the parent.

```python
        new_node.keys = child.keys[t:]
        new_node.avl_trees = child.avl_trees[t:]
        new_node.n = t - 1
```
The RIGHT HALF (keys after the median) goes into new_node.
With t=3: keys[3:] and keys[4:] go to new_node → 2 keys = t-1 keys.

```python
        if not child.is_leaf:
            new_node.children = child.children[t:]
            child.children = child.children[:t]
```
If splitting an internal node, the children must also be divided.
Left child keeps first t children; right child takes the remaining t.

```python
        child.keys = child.keys[:t - 1]
        child.avl_trees = child.avl_trees[:t - 1]
        child.n = t - 1
```
LEFT HALF: original child keeps only keys before the median.
With t=3: keys[:2] → 2 keys = t-1 keys.

```python
        parent.keys.insert(i, median_key)
        parent.avl_trees.insert(i, median_avl)
        parent.children.insert(i + 1, new_node)
        parent.n += 1
```
The median key is inserted into the parent at position i.
The new right-half node is inserted as child[i+1] (right of the split child).
Parent's key count increases by 1.

---

### _inorder method

```python
    def _inorder(self, node: BTreeNode, result: list):
```
In-order traversal of the B-Tree. For each node, the order is:
visit child[0], then key[0], then child[1], then key[1], ..., then key[n-1], then child[n].

```python
        for i in range(node.n):
            if not node.is_leaf:
                self._inorder(node.children[i], result)
            result.append(node.keys[i])
        if not node.is_leaf:
            self._inorder(node.children[node.n], result)
```
This produces disease names in ALPHABETICAL ORDER because that's exactly
what an in-order traversal of a BST gives you.

---

## 4. avl_tree.py — LINE BY LINE

### AVLNode class

```python
class AVLNode:
    def __init__(self, medicine):
```
One node in the AVL tree. Each node represents ONE medicine.

```python
        self.medicine = medicine
```
The full MedicineNode object with all drug data (name, price, composition, etc.)

```python
        self.key = medicine.name
```
The BST ordering key. Medicines are ordered alphabetically by name.
Example: "Aspirin" < "Ibuprofen" < "Paracetamol"

```python
        self.left = None
        self.right = None
```
Child pointers. `None` means no child in that direction.

```python
        self.height = 1
```
A new node is a leaf → height = 1.
Parent nodes update their height after inserting children.

---

### _get_height method

```python
    def _get_height(self, node):
        return node.height if node else 0
```
Return the height, but return 0 for None (empty subtrees have height 0).
This guard prevents NullPointerException-style errors when nodes have
no children.

---

### _get_balance method

```python
    def _get_balance(self, node):
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
```
Balance Factor = left height - right height.
- BF > 1:  left subtree is too tall → need right rotation (or LR rotation)
- BF < -1: right subtree is too tall → need left rotation (or RL rotation)

---

### _rotate_right method

```python
    def _rotate_right(self, z):
```
z is the UNBALANCED node (BF = +2). Its LEFT child y is the pivot.

The rotation transforms:
```
    z (BF=+2)          y
   / \                / \
  y   T4     →      x   z
 / \                   / \
x   T3               T3   T4
```
T3 moves from being y's right child to being z's left child.
y becomes the new root. z becomes y's right child.

```python
        y = z.left        # y is z's left child — will become new root
        T3 = y.right      # T3 is y's right subtree — will move to z's left
        y.right = z       # z becomes y's right child
        z.left = T3       # T3 moves to z's left
        self._update_height(z)   # z is now lower → update first
        self._update_height(y)   # y is now higher → update second
        return y          # y is the new root of this subtree
```

Why update z before y? Because z is now BELOW y in the tree. You must
compute bottom-up heights — a node's height depends on its children.

---

### _rotate_left method

```python
    def _rotate_left(self, z):
```
Mirror image of _rotate_right. z has BF = -2. Right child y is the pivot.

```
  z (BF=-2)              y
 / \                    / \
T1   y         →       z   x
    / \                / \
   T2   x            T1   T2
```

```python
        y = z.right       # y is z's right child — will become new root
        T2 = y.left       # T2 is y's left subtree — moves to z's right
        y.left = z        # z becomes y's left child
        z.right = T2      # T2 moves to z's right
        self._update_height(z)
        self._update_height(y)
        return y
```

---

### _insert method

```python
    def _insert(self, node, medicine):
```
Recursive BST insert + AVL rebalancing. Returns the new root of this subtree.

```python
        if node is None:
            return AVLNode(medicine)
```
Base case: empty subtree → create new node here.

```python
        if medicine.name < node.key:
            node.left = self._insert(node.left, medicine)
        elif medicine.name > node.key:
            node.right = self._insert(node.right, medicine)
        else:
            node.medicine = medicine
            return node
```
Standard BST insert:
- Less than current key → go left
- Greater than current key → go right
- Equal (duplicate) → OVERWRITE the existing medicine data (no duplicate nodes)

```python
        self._update_height(node)
```
After inserting in a subtree, the height of THIS node might have changed.
Update it before checking balance.

```python
        balance = self._get_balance(node)
```
Calculate the balance factor at this node to decide if rotation is needed.

```python
        # Case A: Left-Left (LL)
        if balance > 1 and medicine.name < node.left.key:
            return self._rotate_right(node)
```
BF > 1 means left subtree is taller. The new node is in the LEFT subtree
of the left child (left-left position). FIX: single right rotation.

```python
        # Case B: Right-Right (RR)
        if balance < -1 and medicine.name > node.right.key:
            return self._rotate_left(node)
```
BF < -1 means right subtree is taller. New node is right-right. FIX: left rotation.

```python
        # Case C: Left-Right (LR)
        if balance > 1 and medicine.name > node.left.key:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
```
Left subtree taller, but new node is in RIGHT part of left child (zigzag).
FIX: First left-rotate the left child (converts to LL case),
then right-rotate this node.

```python
        # Case D: Right-Left (RL)
        if balance < -1 and medicine.name < node.right.key:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
```
Mirror of LR. Right subtree taller, zigzag pattern.
FIX: First right-rotate right child, then left-rotate this node.

```python
        return node
```
If no rotation needed, return the node unchanged.

---

### _filter_inorder method

```python
    def _filter_inorder(self, node, age_group: str, result: list):
        if node is None:
            return
        self._filter_inorder(node.left, age_group, result)
        if age_group in node.medicine.suitable_for:
            result.append(node.medicine)
        self._filter_inorder(node.right, age_group, result)
```
Standard in-order traversal (left → current → right) but with a FILTER:
only add medicines where the requested age group is in the medicine's
`suitable_for` list. This is O(n) because every node must be visited.

---

## 5. splay_tree.py — LINE BY LINE

### SplayNode class

```python
class SplayNode:
    def __init__(self, key: str):
        self.key = key           # disease name string
        self.access_count = 1   # how many times this disease was queried
        self.left = None
        self.right = None
```
Simple BST node. The `access_count` is extra data used for the MRU display.

---

### access method

```python
    def access(self, key: str):
```
The main operation: access (or insert) a disease and splay it to the root.

```python
        if self.root is None:
            self.root = SplayNode(key)
            return
```
Empty tree → just create the first node.

```python
        self.root = self._splay(self.root, key)
```
After splaying, the node with `key` (or the closest node if key doesn't exist)
is brought to the root.

```python
        if self.root.key == key:
            self.root.access_count += 1
```
If the node was found, increment its query counter.

```python
        else:
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
```
Key not found. The splay left the CLOSEST node at root.
We insert the new key as the new root and split the existing tree:
- If new key < root's key: new_node.right = old_root, new_node.left = old_root.left, detach old_root.left
- If new key > root's key: new_node.left = old_root, new_node.right = old_root.right, detach old_root.right
This maintains BST ordering.

---

### _splay method (Top-Down Splay)

```python
    def _splay(self, root, key: str):
```
This is the heart of the Splay Tree. We use TOP-DOWN splaying which builds
two auxiliary subtrees (left_tree, right_tree) without recursion.

```python
        header = SplayNode("")
        left_tree = header
        right_tree = header
        t = root
```
`header` is a DUMMY NODE used to assemble two auxiliary trees.
`left_tree` and `right_tree` both start pointing to header.
`t` is our current position in the tree (starts at root).

Conceptually, we're building:
- left_tree: all nodes with keys LESS than our target key
- right_tree: all nodes with keys GREATER than our target key
- `t`: the remaining tree we're still processing

```python
        while True:
            if key < t.key:
```
Target is in the LEFT subtree of t.

```python
                if t.left is None:
                    break
```
No left child → key doesn't exist, stop as close as possible.

```python
                if key < t.left.key:
                    t = self._rotate_right(t)
                    if t.left is None:
                        break
```
ZIG-ZIG LEFT: key is smaller than BOTH t AND t.left.
Rotate right (brings t.left up, pushes t down).
This is the key efficiency trick: rotating the grandparent first amortises
the cost of deep accesses.

```python
                right_tree.left = t
                right_tree = t
                t = t.left
                right_tree.left = None
```
LINK RIGHT: t is larger than our key, so it belongs in right_tree.
We attach t to the bottom of right_tree, then advance t to t.left.
`right_tree.left = None` detaches t's left pointer (we'll set it at end).

```python
            elif key > t.key:
                # Mirror image of the left case
                if t.right is None:
                    break
                if key > t.right.key:
                    t = self._rotate_left(t)
                    if t.right is None:
                        break
                left_tree.right = t
                left_tree = t
                t = t.right
                left_tree.right = None
```
Symmetric case for the right subtree.

```python
            else:
                break
```
FOUND: t.key == key. Stop the loop.

```python
        left_tree.right = t.left
        right_tree.left = t.right
        t.left = header.right
        t.right = header.left
        return t
```
ASSEMBLE: Reconnect the three parts.
- left_tree's rightmost spot gets t's left subtree
- right_tree's leftmost spot gets t's right subtree
- t's left gets everything in the left auxiliary tree
- t's right gets everything in the right auxiliary tree
- t is now the root

This is O(1) final assembly after the loop.

---

## 6. fibonacci_heap.py — LINE BY LINE

### FibNode class

```python
class FibNode:
    def __init__(self, key: float, value):
        self.key = key      # negated score (-actual_score)
        self.value = value  # the MedicineNode object
        self.degree = 0     # number of children
        self.mark = False   # True if lost one child since becoming a child
        self.parent = None  # None if this node is in the root list
        self.child = None   # pointer to one child (circular doubly linked list)
        self.left = self    # left sibling (self-loop for new singleton)
        self.right = self   # right sibling (self-loop for new singleton)
```

`self.left = self` and `self.right = self` creates a self-loop, meaning
a new node is already in a valid circular doubly-linked list of size 1.
This avoids special-casing singletons in the list operations.

`mark`: Used in decrease_key. When a non-root node loses a child,
it gets marked. If it loses a SECOND child (already marked), it gets
cut from its parent and moved to the root list. This ensures no node
loses more than one child, keeping tree degrees bounded at O(log n).

---

### insert_max and _insert

```python
    def insert_max(self, score: float, medicine):
        self._insert(-score, medicine)
```
We NEGATE the score so the internal min-heap gives us the MAXIMUM scorer.
Score 0.92 becomes -0.92 internally. The minimum of all negated scores
is the maximum of all actual scores.

```python
    def _insert(self, key: float, value):
        node = FibNode(key, value)
        self._add_to_root_list(node)
        if self.min_node is None or node.key < self.min_node.key:
            self.min_node = node
        self.n += 1
        return node
```
1. Create the node
2. Add to root list (O(1) — just pointer updates)
3. Update min pointer if this new node has a smaller key
4. Increment count

**Why O(1)?** No sorting, no tree restructuring. The Fibonacci Heap is
"lazy" — it defers all work to extract_min.

---

### _extract_min method

```python
    def _extract_min(self):
        z = self.min_node
        if z is None:
            return None
```
Empty heap guard.

```python
        if z.child is not None:
            children = self._get_siblings(z.child)
            for child in children:
                self._add_to_root_list(child)
                child.parent = None
```
STEP 1: Promote all children of the minimum node to the root list.
Why? When we remove z, its children need a new home. The root list
is where all "top-level" trees live in a Fibonacci Heap.

```python
        self._remove_from_root_list(z)
        self.n -= 1
```
STEP 2: Remove z from the root list and decrement count.

```python
        if z == z.right:
            self.min_node = None
```
If z was the only node in the root list, heap is now empty.

```python
        else:
            self.min_node = z.right
            self._consolidate()
```
Otherwise, temporarily set min to z's right neighbour, then consolidate
to fix the heap and find the true new minimum.

---

### _consolidate method

```python
    def _consolidate(self):
```
After extract_min, the root list may have many trees of the same degree.
Consolidate merges trees of equal degree until all degrees are unique.

```python
        max_degree = int(math.log(self.n + 1) / math.log(1.618)) + 2 if self.n > 0 else 1
        A = [None] * (max_degree + 1)
```
Create an array A indexed by DEGREE. `A[d]` will hold the unique tree of degree d.
Maximum degree is O(log_φ(n)) where φ = (1+√5)/2 ≈ 1.618.
Why 1.618 (golden ratio)? The Fibonacci Heap gets its name from the fact that
a tree of degree k has at least F(k+2) nodes, where F is the Fibonacci sequence.
Since F(k) grows like φ^k, the max degree is log_φ(n).

```python
        roots = self._get_siblings(self.min_node)
```
Snapshot of all current root nodes (we'll modify the list during consolidation).

```python
        for w in roots:
            x = w
            d = x.degree
            while A[d] is not None:
                y = A[d]
                if x.key > y.key:
                    x, y = y, x
                self._heap_link(y, x)
                A[d] = None
                d += 1
            A[d] = x
```
For each root w:
- While there's already a tree of the same degree d:
  - Make the one with LARGER key (= lower score) a child of the other
    (the one with SMALLER key = higher score becomes the root)
  - Clear A[d] (this slot is now empty)
  - d += 1 (the merged tree has one more degree)
- Place x at A[d]

After this loop, all degrees are unique.

```python
        self.min_node = None
        for node in A:
            if node is not None:
                node.left = node
                node.right = node
                if self.min_node is None:
                    self.min_node = node
                else:
                    self._add_to_root_list(node)
                    if node.key < self.min_node.key:
                        self.min_node = node
```
Rebuild the root list from A and find the new minimum.

---

### _heap_link method

```python
    def _heap_link(self, y: FibNode, x: FibNode):
```
Make y a child of x. Precondition: x.key <= y.key.

```python
        self._remove_from_root_list(y)
        y.parent = x
```
Remove y from root list, point it to its new parent x.

```python
        if x.child is None:
            x.child = y
            y.left = y
            y.right = y
        else:
            self._add_to_child_list(x, y)
```
Add y to x's circular child list.

```python
        x.degree += 1
        y.mark = False
```
x now has one more child. y is a fresh child — unmark it.

---

### _decrease_key and _cascading_cut

```python
    def _decrease_key(self, x: FibNode, new_key: float):
        x.key = new_key
        y = x.parent
        if y is not None and x.key < y.key:
            self._cut(x, y)
            self._cascading_cut(y)
        if x.key < self.min_node.key:
            self.min_node = x
```
If we decrease a key and it now violates heap order (child smaller than parent):
1. Cut x from its parent → move to root list
2. Cascading cut: if y was already marked (lost a child before), cut y too,
   and propagate upward

```python
    def _cascading_cut(self, y: FibNode):
        z = y.parent
        if z is not None:
            if not y.mark:
                y.mark = True    # first child loss → mark it
            else:
                self._cut(y, z)          # second child loss → cut it
                self._cascading_cut(z)   # propagate upward
```
This is what keeps the tree degree bounded. Each node can lose at most
one child before it itself gets cut. This ensures trees stay "fat"
(degree proportional to log n).

---

### Circular doubly linked list helpers

```python
    def _add_to_root_list(self, node: FibNode):
        if self.min_node is None:
            node.left = node
            node.right = node
        else:
            node.right = self.min_node.right
            node.left = self.min_node
            self.min_node.right.left = node
            self.min_node.right = node
```
Insert node to the RIGHT of min_node. Four pointer updates:
1. node.right = min_node's old right neighbour
2. node.left = min_node
3. min_node's old right neighbour's left = node
4. min_node.right = node

This is O(1) regardless of list size — no traversal needed.

```python
    def _remove_from_root_list(self, node: FibNode):
        node.left.right = node.right
        node.right.left = node.left
```
Bypass node in both directions. O(1).

```python
    def _get_siblings(self, start: FibNode) -> list:
        nodes = []
        current = start
        while True:
            nodes.append(current)
            current = current.right
            if current == start:
                break
        return nodes
```
Traverse the circular list until we come back to `start`. O(k) where k is list size.

---

## 7. recommendation_engine.py — LINE BY LINE

### MedicineNode class

```python
class MedicineNode:
    def __init__(self, name, disease_target, composition,
                 suitable_for, price, effectiveness_score, availability):
```
This is the DATA MODEL for a single medicine. It's not a data structure
itself — it's the payload stored inside AVL Tree nodes and Fibonacci Heap nodes.

```python
        self.name = name                         # e.g. "Paraflu XR"
        self.disease_target = disease_target     # e.g. "Seasonal Flu"
        self.composition = composition           # list of {"ingredient","mg","percentage"}
        self.suitable_for = suitable_for         # ["Child","Adult","Senior"] subset
        self.price = float(price)                # e.g. 145.50
        self.effectiveness_score = float(effectiveness_score)  # 0.0 to 1.0
        self.availability = bool(availability)   # True if in stock
```

---

### NLP Layer — _SPELL dictionary

```python
_SPELL = {
    "vomitting": "vomiting",
    "threw up": "vomiting",
    "throwing up": "vomiting",
    ...
}
```
**Layer 1 — Spell Correction.** A lookup table mapping incorrect or variant
spellings to their canonical form. Entries are sorted by LENGTH (longest first)
during processing so multi-word phrases ("throwing up and loose motions") are
matched BEFORE their sub-phrases ("throwing up").

This fixes the original `vomitting` bug: the word is looked up in _SPELL,
replaced with `"vomiting"`, and then subsequent layers can match it correctly.

---

### _SYN dictionary

```python
_SYN = {
    "tummy": "stomach",
    "belly": "stomach",
    "ocular": "eye",
    "hurts": "pain",
    "itchy": "itching",
    ...
}
```
**Layer 2 — Synonym Expansion.** Single-word replacements.
Applied WORD BY WORD to the input after spell correction.
Example: "my tummy hurts" → "my stomach pain"
Now "stomach pain" is a phrase in _PHRASES → votes for Gastritis.

---

### _SUFFIX_RULES list

```python
_SUFFIX_RULES = [
    (_re.compile(r'\b(\w+?)itis\b', _re.I), r'\1 inflammation'),
    (_re.compile(r'\b(\w+?)algia\b', _re.I), r'\1 pain'),
    (_re.compile(r'\b(\w+?)opathy\b', _re.I), r'\1 disease'),
    (_re.compile(r'\b(\w+?)uria\b', _re.I), r'urine \1'),
    ...
]
```
**Layer 3 — Suffix Stripping.** Regex patterns that catch ANY medical word
ending in a known suffix. This is powerful because it works on words NOT
in the dictionary:
- "urethritis" → "urethra inflammation" (the root "urethr" becomes "urethra")
- "arthralgia" → "arthr pain" → after synonym: "joint pain"
- "hematuria" → "urine hemat" → combined with context → UTI

The `(\w+?)` capture group is NON-GREEDY — it matches as few characters
as possible, giving us the shortest valid root.

---

### _JARGON dictionary

```python
_JARGON = {
    "dyspnea": "difficulty breathing breathlessness copd exertion",
    "tachycardia": "heart racing fast palpitations arrhythmia",
    "epistaxis": "nosebleed nose hypertension",
    "pruritus": "itching skin eczema",
    ...
}
```
**Layer 4 — Jargon Translation.** Maps Latin/Greek medical terms to plain
English phrases that then match entries in _PHRASES or _PATTERNS.
The translation expands to MULTIPLE words to give the voting system
more chances to accumulate score for the correct disease.

Example: "tachycardia" → "heart racing fast palpitations arrhythmia"
This phrase directly contains "palpitations" which is a phrase in _PHRASES → Arrhythmia.

---

### _PHRASES dictionary

```python
_PHRASES = {
    "nausea vomiting": "Gastritis",
    "vomiting": "Gastritis",
    "nausea": "Gastritis",
    "stomach pain": "Gastritis",
    "acid reflux": "GERD",
    "heartburn": "GERD",
    ...
}
```
**Layer 5 — Phrase Dictionary.** 500+ direct phrase-to-disease mappings.
Sorted longest-first during matching. WEIGHT = number of words in phrase:
- "nausea and vomiting" (3 words) → 3 votes for Gastritis
- "vomiting" (1 word) → 1 vote for Gastritis

Longer phrases are more specific and get higher weight. This prevents
a single generic word ("pain") from overwhelming a specific phrase ("joint pain").

---

### _PATTERNS list

```python
_PATTERNS = [
    (_re.compile(r'\b(vomit|vomiting|vomitting|threw up)\b', _re.I), "Gastritis"),
    (_re.compile(r'\bchest\b.{0,15}\btight\b.{0,20}\b(run|running)\b', _re.I), "Asthma"),
    ...
]
```
**Layer 6 — Regex Sentence Patterns.** Each pattern is a compiled regular expression
that matches natural language input. Weight = 3 (highest of any layer).

`.{0,15}` means "up to 15 any characters" — this allows stop words like "feels",
"is", "really" to appear between the key words without breaking the match.

Example patterns:
- `\bchest\b.{0,15}\btight\b.{0,20}\b(run|running)\b` matches:
  "chest feels tight when i run" → votes 3 for Asthma
- `\b(vomit|vomitting|throwing up)\b` matches any variant → votes 3 for Gastritis

---

### symptoms_to_disease function

```python
def symptoms_to_disease(symptom_string: str) -> str:
    scores: dict = {}
    def _vote(disease: str, pts: float):
        scores[disease] = scores.get(disease, 0.0) + pts
```
`scores` accumulates total vote points per disease.
`_vote` is a closure — it captures `scores` from the outer function's scope.

```python
    raw = symptom_string.lower().strip()
    raw = _re.sub(r"[^\w\s\-']", ' ', raw)
    raw = _re.sub(r'\s+', ' ', raw)
```
L0: Normalise. Lowercase, strip punctuation (keep hyphens/apostrophes),
collapse multiple spaces into one.

```python
    corrected = raw
    for phrase in sorted(_SPELL.keys(), key=len, reverse=True):
        if phrase in corrected:
            corrected = corrected.replace(phrase, _SPELL[phrase])
```
L1: Spell correction. Longest phrase first to avoid partial re-matches.

```python
    def _expand(text):
        return ' '.join(_SYN.get(w, w) for w in text.split())
    expanded_raw = _expand(raw)
    expanded_cor = _expand(corrected)
```
L2: Synonym expansion. Each word independently looked up in _SYN.
If not found, word is kept as-is.

```python
    def _strip_suffixes(text):
        for pattern, repl in _SUFFIX_RULES:
            text = pattern.sub(repl, text)
        return text
    suffix_raw = _strip_suffixes(raw)
    suffix_cor = _strip_suffixes(corrected)
```
L3: Apply all suffix rules in sequence to raw and corrected text.

```python
    def _translate_jargon(text):
        words = text.split()
        out = []
        i = 0
        while i < len(words):
            if i + 1 < len(words) and words[i]+' '+words[i+1] in _JARGON:
                out.append(_JARGON[words[i]+' '+words[i+1]])
                i += 2
                continue
            out.append(_JARGON.get(words[i], words[i]))
            i += 1
        return ' '.join(out)
```
L4: Jargon translation. Try 2-word entries first (longer = more specific),
then single word. `i += 2` skips the second word after a 2-word match.

```python
    variants = list(dict.fromkeys([
        raw, corrected, expanded_raw, expanded_cor,
        suffix_raw, suffix_cor, jargon_raw, jargon_suf, jargon_cor,
        _expand(suffix_raw), _expand(suffix_cor),
        _expand(jargon_raw), _expand(jargon_suf),
    ]))
```
Build a deduplicated list of all text variants. `dict.fromkeys()` preserves
order while removing duplicates. We search EVERY variant in every layer,
maximising the chance that some transformation unlocks a match.

```python
    for variant in variants:
        for pattern, disease in _PATTERNS:
            if pattern.search(variant):
                _vote(disease, 3)
```
L6: Regex patterns. `re.search` checks if the pattern exists ANYWHERE in
the variant string (not just at the start like `re.match`).
Weight 3 — regex patterns are the most specific and reliable signals.

```python
    for variant in variants:
        for phrase in _PHRASES_SORTED:
            if phrase in variant:
                pts = max(1, len(phrase.split()))
                _vote(_PHRASES[phrase], pts)
```
L5: Phrase dictionary. `in` is Python substring search — O(len(variant)).
Weight = word count of phrase (min 1).

```python
    all_tokens = set()
    for v in variants:
        all_tokens.update(v.split())
    all_tokens = {t for t in all_tokens if len(t) > 3}
    for disease in _ENGINE_DISEASE_SET:
        d_tokens = set(disease.lower().split())
        overlap = all_tokens & d_tokens
        if overlap:
            _vote(disease, len(overlap) * 0.5)
```
L7: Token overlap. Split all variants into individual words, filter short
words (> 3 chars avoids "and", "the", "for"). For each disease, count
how many of its name's words appear in the query. 0.5 points per match.
This catches "kidney" → "Chronic Kidney Disease" even without an explicit phrase entry.

```python
    if not scores:
        return "Seasonal Flu"
    return max(scores, key=scores.get)
```
If nothing voted, fall back to Seasonal Flu. Otherwise return the disease
with the highest accumulated score.

---

### _composition_match function

```python
def _composition_match(med_a: MedicineNode, med_b: MedicineNode) -> float:
    dict_a = {c["ingredient"].lower(): c["mg"] for c in med_a.composition}
    dict_b = {c["ingredient"].lower(): c["mg"] for c in med_b.composition}
```
Convert compositions to dictionaries for O(1) lookup by ingredient name.
Using `.lower()` for case-insensitive matching.

```python
    total = max(len(dict_a), len(dict_b))
```
Normalise by the LARGER of the two ingredient lists. This penalises
candidates that have extra ingredients not in the baseline.

```python
    for ing, mg_b in dict_b.items():
        if ing not in dict_a:
            continue
        mg_a = dict_a[ing]
        if mg_b == 0:
            score += 1.0
        else:
            ratio = abs(mg_a - mg_b) / mg_b
            if ratio <= 0.01:   score += 1.0   # exact match
            elif ratio <= 0.20: score += 0.5   # close match
            else:               score += 0.2   # wrong dose
    return score / total
```
For each ingredient in the BASELINE, check if the candidate has it
and how close the dose is. 1% tolerance for "exact" handles floating-point rounding.

---

### score_path_a and score_path_b

```python
def score_path_a(candidate, baseline, max_price):
    comp   = _composition_match(candidate, baseline)
    eff    = candidate.effectiveness_score
    pen    = abs(candidate.price - baseline.price) / max_price
    avail  = 1.0 if candidate.availability else 0.0
    return max(0.0, comp*0.40 + eff*0.30 - pen*0.20 + avail*0.10)
```
Weighted linear formula:
- 40% composition similarity (most important — same ingredients)
- 30% clinical effectiveness
- -20% price difference penalty (normalised against max price)
- +10% availability bonus

`max_price` normalisation ensures the price penalty is in [0,1].
`max(0.0, ...)` clamps to non-negative (a very expensive unavailable drug won't score negative).

```python
def score_path_b(candidate, max_price):
    eff        = candidate.effectiveness_score
    avail      = 1.0 if candidate.availability else 0.0
    norm_price = candidate.price / max_price
    return max(0.0, eff*0.50 + avail*0.30 - norm_price*0.20)
```
Path B has NO baseline, so composition match is removed.
Effectiveness is weighted more (50%) because it's the primary quality signal.
Availability is weighted more (30%) since no baseline context means practicality matters more.

---

### RecommendationEngine class

```python
class RecommendationEngine:
    def __init__(self, t: int = 3):
        self.db = BTree(t=t)
        self.mru_cache = SplayTree()
        self._all_medicines: list = []
        self._disease_set: set = set()
        self._medicine_keys: set = set()
```
The engine is a FACADE (design pattern) — it hides the complexity of
four data structures behind a simple interface.

- `db`: B-Tree storing disease→AVLTree mappings
- `mru_cache`: Splay Tree tracking which diseases are queried most
- `_all_medicines`: flat list for price normalisation calculations
- `_disease_set`: Python set for O(1) "is this disease known?" checks
- `_medicine_keys`: set of (name, disease) tuples to prevent duplicates

---

### load_csv method

```python
    def load_csv(self, filepath: str):
        import recommendation_engine as _self_module
        ...
        _self_module._ENGINE_DISEASE_SET = self._disease_set
```
After loading, we inject the disease set into the MODULE-LEVEL variable
`_ENGINE_DISEASE_SET`. This is used by `symptoms_to_disease` (Layer 7)
which runs at module level. We use the module itself as a namespace.

---

### find_alternatives method

```python
    def find_alternatives(self, disease, baseline_name, age_group, top_n):
        avl = self._get_avl(disease)          # O(log n) B-Tree + O(log n) Splay
        baseline = avl.search(baseline_name)  # O(log k) AVL search
        candidates = [c for c in avl.filter_by_age_group(age_group)
                      if c.name != baseline_name]  # O(k) traversal
        max_price = max(c.price for c in candidates)
        heap = FibonacciHeap()
        for med in candidates:
            heap.insert_max(score_path_a(med, baseline, max_price), med)  # O(1) each
        results = []
        for _ in range(min(top_n, heap.size())):
            results.append(...)
            heap.extract_max()  # O(log k) each
```
This is the full pipeline for Path A. The Fibonacci Heap is ideal here
because we do many O(1) inserts and then only top_n O(log n) extractions.

---

### _get_avl method

```python
    def _get_avl(self, disease: str):
        self.mru_cache.access(disease)   # update splay tree MRU cache
        return self.db.search(disease)   # O(log n) B-Tree search
```
Every disease lookup updates the Splay Tree. If the same disease is
queried repeatedly (e.g., "Seasonal Flu"), it stays at the Splay Tree root
for near-O(1) access, while the B-Tree is still the authoritative store.

---

## 8. generate_dataset.py — LINE BY LINE

```python
def _smart_round_mg(value: float) -> float:
```
This function rounds mg values to clinically realistic precision.

```python
    if value <= 0:    return 0.001    # safety guard
    if value < 0.01:  return round(value, 4)  # e.g. 0.005mg Latanoprost
    if value < 1.0:   return round(value, 3)  # e.g. 0.125mg Digoxin
    if value < 20.0:  return round(value*2)/2  # nearest 0.5mg
    if value < 100.0: return round(value)      # nearest 1mg
    return round(value/5)*5                    # nearest 5mg
```
Why different tiers? Clinical doses have standard rounding conventions:
- Sub-microgram drugs (eye drops, potent cardiac drugs) need 4 decimal places
- Sub-milligram drugs (Digoxin 0.125mg) need 3 decimal places  
- Small doses (1-20mg) round to 0.5mg increments
- Medium doses (20-100mg) round to 1mg
- Large doses (100mg+) round to 5mg tablet sizes

**This fixed the original bug where `round(1.5/5)*5 = 0`.**

---

```python
def generate_composition(template_ingredients):
    mg_values = []
    for ing in template_ingredients:
        mg_min, mg_max = ing["mg_range"]
        raw = random.uniform(mg_min, mg_max)
        mg_values.append(_smart_round_mg(raw))

    total_mg = sum(mg_values)
    composition = []
    running_pct = 0.0
    for idx, (ing, mg) in enumerate(zip(template_ingredients, mg_values)):
        if idx < n - 1:
            pct = round(mg / total_mg * 100, 1)
            running_pct += pct
        else:
            pct = round(100.0 - running_pct, 1)   # residual absorption
        composition.append({"ingredient": ing["ingredient"], "mg": mg, "percentage": pct})
    return composition
```
**This fixed the percentage bug.** Percentages are now derived from actual mg values,
not randomly assigned. The last ingredient absorbs any floating-point residual
(e.g., if previous percentages sum to 99.9 due to rounding, last gets 0.1
to ensure the total is exactly 100.0).

---

## 9. ui_app.py — LINE BY LINE

```python
import tkinter as tk
from tkinter import ttk, messagebox
import threading
```
`tkinter` is Python's standard GUI library (wraps the Tcl/Tk framework).
`ttk` provides themed widgets (Combobox, Notebook, Scrollbar).
`threading` allows the engine to load in a background thread so the UI
doesn't freeze during the 1-2 second CSV loading time.

---

```python
class DrugRecommenderApp(tk.Tk):
```
We inherit from `tk.Tk` to make the app itself the root Tkinter window.
This is the simplest pattern for single-window Tkinter apps.

---

```python
    def _load_engine(self):
        try:
            if not os.path.exists(DATASET_PATH):
                generate_dataset(DATASET_PATH, 2200)
            eng = RecommendationEngine(t=3)
            eng.load_csv(DATASET_PATH)
            self.engine = eng
            self.engine_ready = True
            self.after(0, self._on_engine_ready)
        except Exception as e:
            self._set_status(f"✗  Error: {e}", "rose")
```
Runs in a BACKGROUND THREAD. `self.after(0, callback)` schedules the
callback to run in the MAIN THREAD — this is mandatory because Tkinter
is NOT thread-safe. You must never update widgets from a background thread.

`self.after(0, fn)` means "run fn as soon as the main event loop is free"
with a delay of 0 milliseconds.

---

```python
    def _exec_path_a(self, top_n):
        ...
        def _worker():
            res = self.engine.find_alternatives(disease, medicine, age, top_n)
            title = f"Alternatives for '{medicine}'  ·  {disease}  ·  {age}"
            self.after(0, lambda: self._display_results(res, title))
            self.after(0, self._update_mru)
        self._show_loading()
        threading.Thread(target=_worker, daemon=True).start()
```
The query also runs in a background thread (queries can take ~100ms
for large trees). `daemon=True` means the thread is killed automatically
when the main app closes (no orphan threads).

---

```python
    def _build_card(self, parent, rank, item):
```
Builds one result card. Each card is a `tk.Frame` with:
- A 4px coloured accent bar on the left (gold for #1, silver for #2, etc.)
- Medicine name and availability badge
- Score bar (coloured green/amber/red by score threshold)
- Composition list with mg and percentage
- Price, effectiveness percentage, demographic badges

```python
        bar_bg = tk.Frame(srow, bg=C["border"], height=6, width=220)
        bar_bg.pack_propagate(False)
        fill_w = max(2, int(score * 220))
        tk.Frame(bar_bg, bg=bar_clr, width=fill_w, height=6).pack(side="left")
```
The score bar is a fixed-width grey frame (`bar_bg`) with a coloured
inner frame (`fill_w` pixels wide) packed to the left.
`pack_propagate(False)` prevents the outer frame from shrinking to fit its contents.
`score * 220` converts [0,1] score to pixel width [0,220].
`max(2, ...)` ensures even a 0-score shows a minimal visible bar.

---

```python
    def _on_canvas_resize(self, e):
        self.canvas.itemconfig(self._cwin, width=e.width)
```
When the right panel is resized, this makes the scrollable frame
expand to fill the full width. Without this, cards would be narrow
even in a wide window.

---

```python
    def _clear_placeholder(self, _):
        if self.sym_text.get("1.0", "end-1c") == self._placeholder_text:
            self.sym_text.delete("1.0", "end")
            self.sym_text.config(fg=C["text"])
    def _restore_placeholder(self, _):
        if not self.sym_text.get("1.0", "end-1c").strip():
            self.sym_text.config(fg=C["text_sub"])
            self.sym_text.insert("1.0", self._placeholder_text)
```
Placeholder text pattern: show grey hint text when empty, clear on focus,
restore on blur if still empty. `"1.0"` in Tkinter Text widget means
"line 1, character 0". `"end-1c"` means "end minus 1 character"
(to exclude the automatic trailing newline that Text always adds).

---

## 10. HOW ALL FILES CONNECT TOGETHER

```
generate_dataset.py
    │  creates
    ▼
medicines_dataset.csv  ──────────────────────────────────────────┐
                                                                  │
ui_app.py                                                         │
    │  imports and creates                                         │
    ▼                                                             │
RecommendationEngine (recommendation_engine.py)                   │
    │  calls load_csv() ───────────────────────────────────────────┘
    │
    │  for each row:
    │      ┌─── creates MedicineNode ──────── (recommendation_engine.py)
    │      │
    │      └─── calls BTree.insert_disease() ── (b_tree.py)
    │               │
    │               └─── which internally calls AVLTree.insert() ── (avl_tree.py)
    │
    │  on query:
    │      ├─── calls SplayTree.access() ─────── (splay_tree.py)
    │      ├─── calls BTree.search() ──────────── (b_tree.py)
    │      │        └─── returns AVLTree
    │      │                 └─── calls filter_by_age_group()
    │      ├─── calls score_path_a/b() ────────── (recommendation_engine.py)
    │      └─── calls FibonacciHeap.insert_max() / extract_max() ── (fibonacci_heap.py)
    │
    │  for Path B:
    │      └─── calls symptoms_to_disease() ───── (recommendation_engine.py)
    │               (7-layer NLP pipeline)
    │
    └─── results displayed in ui_app.py cards
```

---

## 11. TIME COMPLEXITY SUMMARY

| Operation | Data Structure | Time Complexity | Why |
|---|---|---|---|
| Insert disease | B-Tree | O(t · log_t n) | Traverse height, split O(t) nodes |
| Search disease | B-Tree | O(t · log_t n) | Linear scan within nodes × height |
| Insert medicine | AVL Tree | O(log k) | BST insert + at most 2 rotations |
| Search medicine | AVL Tree | O(log k) | Standard BST search |
| Age filter | AVL Tree | O(k) | Must visit all k nodes |
| Disease access | Splay Tree | O(log n) amortised | Splay to root |
| MRU query | Splay Tree | O(1) | Just return root |
| Insert score | Fibonacci Heap | O(1) | Lazy — no restructuring |
| Get top recommendation | Fibonacci Heap | O(log m) | Extract min + consolidate |
| Symptom mapping | NLP Mapper | O(P + V·R) | P phrases, V variants, R regex patterns |
| Full Path A query | Combined | O(t·log_t n + log k + k + m·log m) | n diseases, k medicines, m candidates |
| Full Path B query | Combined | O(NLP + t·log_t n + k + m·log m) | same + NLP mapping |

Where:
- n = total number of diseases (61 in this system)
- k = medicines per disease (average ~36)
- m = candidates after age filter (subset of k)
- t = B-Tree minimum degree (3)
- P = number of phrases (~500)
- V = number of text variants generated (~13)
- R = number of regex patterns (~120)

---

## 12. EVERY CONCEPT EXPLAINED — PLAIN LANGUAGE GLOSSARY

This section explains every single CS concept used in the project as if
you are hearing it for the first time.

---

### 12.1 What is a Tree?

A TREE is a way of organising data in a hierarchy, like a family tree
or a company org chart. It has:

- A ROOT: the single node at the top (like the CEO of a company)
- NODES: individual elements that hold data
- EDGES: connections between nodes (parent → child)
- LEAVES: nodes with no children (the bottom of the hierarchy)
- HEIGHT: the number of levels from root to the deepest leaf

In this project, trees are used everywhere because they allow
FAST SEARCHING — instead of checking every item one by one (O(n)),
trees allow you to eliminate half (or more) of the options at every
step (O(log n)).

---

### 12.2 What is a Binary Search Tree (BST)?

A BST is a tree where:
- Each node has AT MOST 2 children: left and right
- LEFT subtree only contains values SMALLER than the node
- RIGHT subtree only contains values LARGER than the node

Searching works like a phone book:
- Is the name I want before or after the current page?
- Go left (earlier) or right (later)
- Repeat until found

Example searching for "Ibuprofen" in this tree:
```
           Metformin
          /         \
    Ibuprofen      Paracetamol
```
- "Ibuprofen" < "Metformin"? YES → go left
- "Ibuprofen" == "Ibuprofen"? YES → FOUND

WHERE USED: AVL Tree and Splay Tree are both BSTs with extra self-balancing rules.

---

### 12.3 What is Tree Balance and Why Does It Matter?

Imagine a BST where you insert names in alphabetical order:
```
Aspirin
   \
  Ibuprofen
        \
       Metformin
              \
             Paracetamol
```
This is a SKEWED TREE — it's just a linked list in disguise.
Searching for Paracetamol requires visiting ALL 4 nodes: O(n) not O(log n).

A BALANCED tree keeps the height at O(log n) by ensuring no branch is
much taller than another. For 1000 medicines, a balanced tree has height
~10 (log₂ 1000 ≈ 10), so at most 10 comparisons to find any medicine.

An unbalanced tree could have height 1000 → 1000 comparisons.

WHERE USED: The AVL Tree automatically rebalances after every insert using rotations.

---

### 12.4 What is a Rotation?

A rotation is like a "pivot" — it restructures 3 nodes to change the
height distribution WITHOUT breaking the BST ordering rule.

RIGHT ROTATION (imagine physically rotating the triangle clockwise):
```
BEFORE:           AFTER:
    z               y
   /               / \
  y         →     x   z
 /
x
```
Node y "rises" to the top. z "sinks" to the right.
The BST property is preserved: x < y < z, both before and after.

LEFT ROTATION (anti-clockwise):
```
BEFORE:           AFTER:
  z                 y
   \               / \
    y      →      z   x
     \
      x
```

LR ROTATION (Left-Right — two rotations):
```
BEFORE:           STEP 1 (left rotate y):     STEP 2 (right rotate z):
    z                    z                           x
   /                    /                           / \
  y             →      x              →            y   z
   \                  /
    x                y
```

RL ROTATION is the mirror image of LR.

WHERE USED: Every AVL Tree insert checks if rotation is needed and
applies one of these four cases. Each rotation is O(1) — just 2-3 pointer updates.

---

### 12.5 What is a B-Tree and Why is it Different from a BST?

A regular BST has ONE key per node. A B-TREE has MULTIPLE keys per node.

Think of a BST like a filing cabinet where each drawer has ONE label.
A B-Tree is like a filing cabinet where each drawer has MULTIPLE labels
and sub-dividers.

B-Tree node with t=3 can hold 2–5 keys:
```
[ Arthritis | Diabetes | Hypertension ]
    ↓            ↓           ↓            ↓
  (A–Ar)     (D–Hy)      (Hy–Z)        (Z+)
```

WHY IS THIS USEFUL? Fewer levels = fewer steps to find something.
With 61 diseases and t=3, the B-Tree is at most 4 levels deep.
A BST with 61 items is at most 6 levels deep (balanced) or 61 (unbalanced).

ALSO: B-Trees keep all leaves at the SAME depth — they grow from the top,
not the bottom. This guarantees worst-case O(log n) for all operations.

WHERE USED: `b_tree.py` — indexes all 61 diseases. Each disease key
points to an AVL Tree of its medicines.

---

### 12.6 What is a Splay Tree and what is MRU Caching?

A CACHE is a fast storage area that holds recently-used items so they
can be accessed instantly without going back to the slower "main" storage.

Your web browser caches images — the second time you visit a website,
images load faster because they're already in the cache.

An MRU (Most Recently Used) Cache keeps track of which items were used
most recently. The most recently used item is always at the TOP.

A SPLAY TREE implements this automatically:
- Every time you access a node, it's "splayed" (moved) to the ROOT
- The root is always the most recently accessed item
- O(1) to retrieve it next time (it's at the root!)

In this project: if a user queries "Seasonal Flu" 5 times in a row,
"Seasonal Flu" stays at the root of the Splay Tree.
The 6th query finds it instantly at the root without traversing the B-Tree.

WHERE USED: `splay_tree.py` — caches recently queried diseases.
The "SPLAY CACHE — RECENT QUERIES" panel in the UI shows this live.

---

### 12.7 What is a Heap?

A HEAP is a special tree structure with ONE rule:
- Every parent is ALWAYS smaller than its children (min-heap)
  OR always larger (max-heap)

This means the MINIMUM (or maximum) is always at the ROOT — accessible in O(1).

Think of a heap as a tournament bracket:
- The winner of each match (smallest number) advances up
- The overall winner (smallest) is at the top

Example min-heap:
```
        1
       / \
      3   2
     / \
    4   5
```
Rule: every parent ≤ its children. 1 ≤ 3 ✓, 1 ≤ 2 ✓, 3 ≤ 4 ✓, 3 ≤ 5 ✓.

WHERE USED: Fibonacci Heap in `fibonacci_heap.py` — ranks candidate
medicines so the highest-scoring one can be extracted first.

---

### 12.8 What makes a Fibonacci Heap special?

A regular binary heap (like Python's `heapq`) has:
- Insert: O(log n)
- Extract min: O(log n)

A FIBONACCI HEAP has:
- Insert: O(1) — just add to a flat list, don't sort yet
- Extract min: O(log n) amortised — do all the sorting work NOW
- Decrease key: O(1) amortised — very important for graph algorithms

The trick is LAZINESS: a Fibonacci Heap delays all hard work until you
actually need the minimum. Until then, it just maintains a loose collection
of trees (the "root list").

Why is it called "Fibonacci"? Because a tree of degree k in a Fibonacci Heap
has at least F(k+2) nodes (Fibonacci numbers). This bounds the maximum degree
at O(log n), which is why extract_min is O(log n) even after lazy accumulation.

WHERE USED: After scoring all candidate medicines, they're all inserted
into the Fibonacci Heap (O(1) each). Then we extract only the top-N
(O(log n) each). This is optimal for "insert many, extract few" patterns.

---

### 12.9 What is Amortised Complexity?

"Amortised" means averaged over a sequence of operations, not per operation.

Real-world analogy: A printer costs £200 and prints 10,000 pages.
The cost per page is £0.02 — amortised.
Some pages might cost more (replacing ink) but on average it's £0.02.

In algorithms:
- Splay Tree: one access might be O(n) if the tree is skewed.
  But any sequence of m accesses costs O(m log n) total.
  Amortised cost per operation = O(log n).

- Fibonacci Heap insert: O(1). Very cheap! But each insert "saves up debt"
  that gets paid during extract_min. Over any sequence of operations,
  the average cost per operation works out to the stated amortised bounds.

WHERE USED: Complexity claims for Splay Tree and Fibonacci Heap
are amortised, not worst-case per operation.

---

### 12.10 What is a Doubly Linked Circular List?

A LINKED LIST is a chain of nodes where each node points to the next:
```
[A] → [B] → [C] → None
```

A DOUBLY LINKED LIST adds a BACKWARD pointer too:
```
None ← [A] ↔ [B] ↔ [C] → None
```

A CIRCULAR doubly linked list connects the ends:
```
↱ [A] ↔ [B] ↔ [C] ↰
```
The last node's right points back to the first, and the first's left points to the last.

WHY CIRCULAR? The Fibonacci Heap root list grows and shrinks during
consolidation. With a circular list:
- Insert at any position: O(1) — just update 4 pointers
- Remove any node: O(1) — just bypass it in both directions
- Traverse all nodes: O(k) — just follow right pointers until you're back at start

WHERE USED: `fibonacci_heap.py` — the root list and each node's child list
are circular doubly linked lists.

---

### 12.11 What is Recursion?

Recursion is when a function calls ITSELF with a smaller version of the problem.

Example: find the height of a tree node:
```
height(node):
    if node is None: return 0          ← base case (stop here)
    return 1 + max(height(node.left),  ← recursive calls
                   height(node.right))
```

Every recursion needs:
1. A BASE CASE that stops the recursion (node is None → return 0)
2. A RECURSIVE CASE that makes progress toward the base case

WHERE USED: `_insert()` in AVL Tree, `_search_node()` in B-Tree,
`_splay()` in Splay Tree, `_cascading_cut()` in Fibonacci Heap,
`symptoms_to_disease()` processing.

---

### 12.12 What is a Hash Set and Hash Map?

A SET is a collection where each item appears at most once.
A MAP (or Dictionary) associates keys with values.

A HASH SET/MAP uses a HASH FUNCTION to convert a key (like "Arthritis")
into an index (like 42) in an array. This gives O(1) average-case lookup
— no comparison, no traversal, just compute the index and read.

WHERE USED:
- `_disease_set: set` — O(1) check "is this disease known?"
- `_medicine_keys: set` — O(1) deduplication check during CSV loading
- `scores: dict` in NLP mapper — O(1) vote accumulation per disease
- `_SYN`, `_JARGON`, `_SPELL` are all Python dicts — O(1) word lookup

---

### 12.13 What is a Lambda / Closure?

A LAMBDA is an anonymous (unnamed) function defined inline:
```python
key=lambda n: n.access_count
```
This means "a function that takes n and returns n.access_count".
Used in sorting: sort by access_count.

A CLOSURE is a function that "remembers" variables from its enclosing scope:
```python
def symptoms_to_disease(symptom_string):
    scores = {}
    def _vote(disease, pts):     ← _vote is a closure
        scores[disease] = ...    ← it sees `scores` from outer scope
```
`_vote` can see and modify `scores` even though `scores` is defined
in the outer function. This avoids passing `scores` as a parameter
every time `_vote` is called.

WHERE USED: Lambda in sorting (Splay Tree `get_top_k`, engine methods).
Closures in `symptoms_to_disease` (`_vote`) and `load_csv` (`_worker` threads).

---

### 12.14 What is Threading?

A THREAD is an independent path of execution within a program.
By default, Python runs on ONE thread — instructions execute one at a time.

When we load 2200 medicines into the B-Tree, it takes ~1 second.
If this runs on the MAIN THREAD, the UI freezes for that second.

By running the engine loading on a BACKGROUND THREAD, the main thread
is free to keep the UI responsive (user can see the window, resize it, etc.)
while loading happens in parallel.

```python
threading.Thread(target=_load_engine, daemon=True).start()
```
- `target`: the function to run in the background
- `daemon=True`: kill this thread when the main program exits (no orphan threads)

IMPORTANT: Tkinter (the UI) is NOT thread-safe. You must NEVER update
a widget from a background thread. Instead, use:
```python
self.after(0, lambda: self.status_lbl.config(text="Ready"))
```
`after(0, fn)` schedules fn to run in the MAIN thread at the next opportunity.

WHERE USED: `ui_app.py` — engine loading and query execution both use background threads.

---

### 12.15 What is Regex (Regular Expressions)?

A REGEX is a pattern that describes a set of strings.

Examples:
- `\b` means "word boundary" (the edge between a word character and a non-word character)
- `\b(vomit|vomiting)\b` matches "vomit" or "vomiting" as whole words (not "vomitory")
- `.{0,15}` means "any character, between 0 and 15 times"
- `[^\w\s]` means "any character that is NOT a word character or whitespace"

In the NLP mapper:
```python
_re.compile(r'\bchest\b.{0,15}\btight\b.{0,20}\b(run|running)\b', _re.I)
```
This matches "chest feels tight when I run" because:
- `\bchest\b` matches "chest"
- `.{0,15}` matches " feels " (7 characters, within limit)
- `\btight\b` matches "tight"
- `.{0,20}` matches " when I " (8 characters, within limit)
- `\b(run|running)\b` matches "run"

`_re.compile()` pre-compiles the pattern into a fast state machine.
This is why we compile once at module load, not on every call.

WHERE USED: `recommendation_engine.py` — the `_PATTERNS` list and `_SUFFIX_RULES`.

---

### 12.16 What is the Facade Design Pattern?

A FACADE is a class that provides a SIMPLE interface to a complex system.

Instead of:
```python
btree = BTree(t=3)
btree.insert_disease("Flu", med)
avl = btree.search("Flu")
candidates = avl.filter_by_age_group("Adult")
heap = FibonacciHeap()
for c in candidates: heap.insert_max(score(c), c)
results = [heap.extract_max() for _ in range(5)]
```

You just call:
```python
engine = RecommendationEngine()
engine.load_csv("medicines_dataset.csv")
results = engine.find_alternatives("Flu", "Paraflu", "Adult", 5)
```

The `RecommendationEngine` class is the FACADE — it hides all the B-Tree,
AVL, Splay, and Fibonacci Heap complexity behind 3 simple methods.

WHERE USED: `recommendation_engine.py` — `RecommendationEngine` class.

---

### 12.17 What is In-Order Traversal?

A traversal visits every node in a tree. The ORDER matters:

- PRE-ORDER: Visit node FIRST, then left subtree, then right subtree
  (used in: Splay Tree `_collect` method — to gather all nodes)

- IN-ORDER: Visit left subtree FIRST, then node, then right subtree
  (used in: AVL Tree `_inorder`, B-Tree `_inorder` — produces SORTED output)

- POST-ORDER: Visit left subtree, right subtree, THEN node
  (used in: deleting trees — delete children before parent)

WHY does in-order give sorted output?
In a BST: left < node < right.
So: visit all smaller keys (left subtree), then this key, then all larger keys (right subtree).
The result is sorted ascending.

WHERE USED:
- `avl_tree._inorder()` → returns medicines in alphabetical name order
- `b_tree._inorder()` → returns disease names alphabetically
- `splay_tree._collect()` uses pre-order for the MRU top-k calculation

---

### 12.18 What is Memoisation / Caching in the NLP System?

The NLP layer generates up to 13 VARIANTS of the input text
(raw, corrected, suffix-stripped, jargon-expanded, and combinations).
It then searches ALL patterns and phrases against ALL variants.

This is NOT memoisation — it's EXHAUSTIVE COVERAGE. By transforming
the input in every possible way and checking all of them, we maximise
the chance that at least one transformation produces a recognisable signal.

The `_PHRASES_SORTED` list is pre-sorted ONCE at module load:
```python
_PHRASES_SORTED = sorted(_PHRASES.keys(), key=lambda p: len(p.split()), reverse=True)
```
This is computed ONCE when the module is imported, not every time
`symptoms_to_disease` is called. This is a form of PRECOMPUTATION —
doing expensive work ahead of time so runtime calls are fast.

Similarly, regex patterns in `_PATTERNS` are PRE-COMPILED:
```python
(_re.compile(r'\bvomit\b', _re.I), "Gastritis")
```
`_re.compile()` converts the pattern string into a compiled state machine.
Compiling takes time but searching is then O(n) on text length, not O(n×m).

---

## 13. WHAT EACH FILE DOES — SUMMARY TABLE

| File | Role | Key Classes/Functions | Lines |
|---|---|---|---|
| `b_tree.py` | Disease database index | `BTreeNode`, `BTree` | 211 |
| `avl_tree.py` | Medicine store per disease | `AVLNode`, `AVLTree` | 256 |
| `splay_tree.py` | MRU cache | `SplayNode`, `SplayTree` | 212 |
| `fibonacci_heap.py` | Recommendation ranker | `FibNode`, `FibonacciHeap` | 370 |
| `recommendation_engine.py` | NLP + scoring + engine | `MedicineNode`, `symptoms_to_disease`, `RecommendationEngine` | 1420 |
| `generate_dataset.py` | Mock data creator | `_smart_round_mg`, `generate_composition`, `generate_dataset` | 747 |
| `ui_app.py` | User interface | `DrugRecommenderApp` | 549 |

---

## 14. EVERY BUG FOUND AND HOW IT WAS FIXED

| Bug | Root Cause | Fix | File |
|---|---|---|---|
| `Ergotamine 0mg` | `round(1.5/5)*5 = 0` for small mg ranges | `_smart_round_mg()` with tiered precision, no floor | `generate_dataset.py` |
| Percentages wrong (43.5% vs 61.7%) | Percentages were random, unrelated to mg | Derive `pct = mg/total_mg*100`; last absorbs residual | `generate_dataset.py` |
| Digoxin 0.125mg → 0.5mg | Previous fix added `max(0.5, ...)` floor | Remove floor; use `round(value,3)` for sub-1mg | `generate_dataset.py` |
| `vomitting` → Seasonal Flu | Spell table had `"vomitting": "vomiting"` but the `replace()` would re-match already-replaced text | Sort by length descending before replacing; add combo entries | `recommendation_engine.py` |
| Eye irritation → Seasonal Flu | Phrase map had "eye redness" not "eye irritation" | Body-part × descriptor matrix: `(eye, irritation)` | `recommendation_engine.py` |
| Ulcers → Seasonal Flu | Map had "stomach ulcer" not "ulcers" | Regex: `\bulcers?\b` catches both | `recommendation_engine.py` |
| Urethra inflammation → Seasonal Flu | No "urethra" entry anywhere | Body-part matrix: `(urethra, inflammation)` + suffix stripping of "-itis" | `recommendation_engine.py` |
| Dysphagia → Seasonal Flu | Jargon translated to "difficulty swallowing" but no pattern caught it | Jargon now maps to "difficulty swallowing throat gerd acid reflux" | `recommendation_engine.py` |
| Night sweat → Hyperthyroidism | "sweating" synonym fired more Hyperthyroidism votes | Added explicit night-sweats TB patterns with weight 3 | `recommendation_engine.py` |
| Chest tight running → Angina | "chest" + "tight" pattern matched Angina before Asthma | Added `chest.{15}tight.{20}(run|running)` → Asthma with weight 3 | `recommendation_engine.py` |
| Double query execution | `_exec_path_a` launched two threads | Removed dead lambda-thread; clean single worker | `ui_app.py` |
| `_all_medicines` duplicates on reload | Flat list had no deduplication | `_medicine_keys` set tracks `(name, disease)` pairs | `recommendation_engine.py` |
| Symptom mapper only covered 29/61 diseases | Map never updated when diseases were added | Expanded to 140+ entries, then full 7-layer NLP system | `recommendation_engine.py` |

---

## 15. INTERVIEW / EXAM QUESTIONS THIS PROJECT COVERS

1. **Why use a B-Tree instead of a BST for the disease database?**
   B-Trees have a higher branching factor — each node holds multiple keys.
   This keeps the tree shallower (fewer levels) and is more cache-friendly.
   For 61 diseases and t=3, a B-Tree is at most 4 levels deep.

2. **What is the AVL Tree invariant, and how is it maintained?**
   |balance_factor| ≤ 1 at every node. Maintained by detecting imbalance
   (BF = ±2) after every insert and applying one of four rotation cases.

3. **Why use a Splay Tree for MRU caching instead of a hash map?**
   Splay Tree gives automatically ordered access history. The most recently
   accessed item is always at the root (O(1)). A hash map has O(1) lookup
   but doesn't inherently track recency or provide sorted "top-k" queries.

4. **How does the Fibonacci Heap achieve O(1) insert?**
   It's lazy — new nodes are just appended to the root list without
   any sorting or restructuring. All "balancing" work is deferred to
   extract_min. This amortises the cost over many operations.

5. **How do you simulate a max-heap with a min-heap?**
   Negate all keys: store -score instead of score.
   The minimum of all -scores is the maximum of all scores.
   extract_min gives you the highest-scoring item.

6. **What is cascading cut and why is it needed?**
   When decrease_key moves a node to the root list, it might leave its
   parent "damaged". If the parent already lost one child (marked), it
   gets cut too. This propagates upward, preventing trees from becoming
   too tall and keeping degree bounded at O(log n).

7. **What is the time complexity of the full Path A query?**
   O(t·log_t n) for B-Tree search + O(log k) for baseline lookup +
   O(k) for age filter + O(m) for Fibonacci Heap inserts +
   O(top_n · log m) for extractions = O(k + top_n · log k)
   for typical values.

8. **What is amortised complexity, and when does it matter?**
   Amortised = average cost per operation over a long sequence.
   Matters when individual operations vary wildly in cost but the
   total cost is bounded. Splay Tree and Fibonacci Heap both rely on
   amortised analysis to justify their O(log n) claims.
