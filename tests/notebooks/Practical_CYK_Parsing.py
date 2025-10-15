#!/usr/bin/env python
# coding: utf-8
from tests.notebooks.util import get_ipython

# # Practical: CYK Parsing

# ## Working with Charts

# Familiarise yourself with the following two types of charts that are useful for parsing CFGs.

# In[1]:


from muprocdurham.pcfg import SetChart, DictSetChart, cat_pretty
from muprocdurham.jupyter import no_linebreaks
from muprocdurham import seed_everything

seed_everything(42)
no_linebreaks()


# ### SetChart

# A ``SetChart`` stores sets of elements/symbols in each cell. You can initialise and print it like this:

# In[2]:


chart = SetChart(3)
print(chart.pretty(haxis=True))


# You can access a cell via ``chart[start, end]``, where ``start`` and ``end`` are the index of the left and right boundary of the sub-chart ("triangle") of which the corresponding cell is the top. The following simply adds the ``(start, end)`` pairs to each cell to give you an overview of the indexing.

# In[3]:


for start in range(chart.n):
    for end in range(start + 1, chart.n + 1):
        chart[start, end].add((start, end))
print(chart.pretty(haxis=True))


# **Exercise:** Add the symbol 'A' to the very top cell, the symbol 'B' to the center bottom cell, the symbol 'C' to the cell spanning from 1 to 3, and finally a second symbol 'D' to the very top cell.
# 
# *Hint:* ``chart[start, end]`` gives you access to the corresponding ``set`` in that cell, which you can then manipulate.
# 
# Your result should look as below.

# In[4]:


chart = SetChart(3)
# vvvvvvvvvvvvvvvvvv
# FILL HERE
chart[0, 3].add('A')
chart[1, 2].add('B')
chart[1, 3].add('C')
chart[0, 3].add('D')
# ^^^^^^^^^^^^^^^^^^
print(chart.pretty(haxis=True))


# Finally, to help with orientation, you can also print the level, which corresponds to the length of the *span* (the corresponding subsequence) associated to the cells.

# In[5]:


print(chart.pretty(haxis=True, laxis=True))


# ### DictSetChart

# A ``DictSetChart`` is similar to a ``SetChart``, just that is stores a *dictionary* of sets in each cell. This means that you can store a set of values along with each of the symbols.
# 
# **Exercise:** In the top cell, add the values ``1`` and ``2`` to the symbol 'A' and the value ``3`` to the symbol 'B'.
# 
# *Hint:* ``chart[start, end]`` gives you access to a ``dict`` of ``set``s in that cell, which you can then manipulate.
# 
# Your result should look as below.

# In[6]:


chart = DictSetChart(2)
# vvvvvvvvvvvvvvvvvvvvv
# FILL HERE
chart[0, 2]['A'].add(1)
chart[0, 2]['A'].add(2)
chart[0, 2]['B'].add(3)
# ^^^^^^^^^^^^^^^^^^^^^
print(chart.pretty())


# ## Minimal Example by Hand

# - non-terminals: `{A, B}`
# - terminals: `{a, b}`
# - start symbol: `A`
# - rules:
#   - `A --> A A`
#   - `A --> B A`
#   - `A --> a`
#   - `B --> b`
#   
# Sequence to parse: `a a b b a`

# In[7]:


chart = SetChart(5)
sequence = 'aabba'

# fill base level 1
chart[0, 1].add('A')  # A --> a
chart[1, 2].add('A')  # A --> a
chart[2, 3].add('B')  # B --> b
chart[3, 4].add('B')  # B --> b
chart[4, 5].add('A')  # A --> a

# fill level 2
chart[0, 2].add('A')  # A --> A A (split at 1)
chart[3, 5].add('A')  # A --> B A (split at 4)

# fill level 3
chart[2, 5].add('A')  # A --> B A (split at 3)

# fill level 4
chart[1, 5].add('A')  # A --> A A (split at 2)

# fill level 5
chart[0, 5].add('A')  # A --> A A (split at 1)
chart[0, 5].add('A')  # A --> A A (split at 2)

print(chart.pretty(haxis=True, laxis=True))
print('   ' + '     '.join(sequence))


# ## CYK Parsing

# ### Simple CYK Parser

# We are now equipped for implementing a CYK parser for a context-free grammar (CFG). Below is a template class, which you can complete using the pseudocode from the lecture slides.

# In[8]:


class CFG:
    def __init__(self, non_terminal_rules, terminal_rules, start_symbol):

        # store the non-terminal rules as triplets: A --> B C is stored as ('A', 'B', 'C')
        self.non_terminal_rules = set(tuple(r) for r in non_terminal_rules)  
        # store the terminal rules as pairs: A --> a is stored as ('A', 'a')
        self.terminal_rules = set(tuple(r) for r in terminal_rules)  
        # the start symbol
        self.start_symbol = start_symbol
        
        # the chart is initialised by the init_chart() function called by the parse() function
        self.chart = None
        
        # extracting the non-terminal and terminal symbols from the rules
        self.non_terminal_symbols = set.union(*[{x1, x2, x3} for x1, x2, x3 in self.non_terminal_rules])
        self.terminal_symbols = {y for x, y in self.terminal_rules}
        # make sure the start symbol is in the set of non-terminal symbols
        assert start_symbol in self.non_terminal_symbols, f"Start symbol '{start_symbol}' must be in non-terminals"
    
    def init_chart(self, n):
        # Override this function for using other chart types
        self.chart = SetChart(n)
        
    def parse(self, sequence):
        n = len(sequence)
        self.init_chart(n)
        
        # fill bottom row using terminal rules
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # FILL HERE
        for start, terminal_symbol in enumerate(sequence):
            self.fill_terminal_cell(start, terminal_symbol)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        # fill rows, bottom up, using non-terminal rules
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # FILL HERE
        for level in range(2, n + 1):  # length of sub-sequence
            for start in range(n - level + 1):
                for split in range(start + 1, start + level):
                    end = start + level
                    self.fill_non_terminal_cell(start, split, end)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        # return whether the sequence is valid (i.e. whether the start symbol is in the top cell)
        return self.start_symbol in self.chart[0, n]
    
    def fill_terminal_cell(self, start, symbol):
        # go through all terminal rules to fill cell
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # FILL HERE
        for x, y in self.terminal_rules:
            # if right hand side (terminal) matches, add left hand side (non-terminal) to cell
            if y == symbol:
                self.chart[start, start + 1].add(x)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    def fill_non_terminal_cell(self, start, split, end):
        # go through all non-terminal rules to fill cell
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # FILL HERE
        for x1, x2, x3 in self.non_terminal_rules:
            # if symbols on right hand side are in the corresponding cells,
            # add left hand side to parent cell
            if x2 in self.chart[start, split] and x3 in self.chart[split, end]:
                self.chart[start, end].add(x1)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# Test your implementation using the minimal example from the lecture.

# In[9]:


cfg = CFG(
    non_terminal_rules=[('A', 'A', 'A'), ('A', 'B', 'A')],
    terminal_rules=[('A', 'a'), ('B', 'b')],
    start_symbol='A'
)
print("non-terminal rules:", cfg.non_terminal_rules)
print("terminal rules:", cfg.terminal_rules)
print("start symbol:", cfg.start_symbol)
print("non-terminal symbols:", cfg.non_terminal_symbols)
print("terminal symbols:", cfg.terminal_symbols)

print("valid sequence:", cfg.parse("aabba"))
print(cfg.chart.pretty())


# ### Getting Parse Trees

# We can now extend our implementation to also remember splitting points, so we can later reconstruct all possible parse trees.
# 
# **Exercise:** Override the ``fill_terminal_cell`` and ``fill_non_terminal_cell`` functions to remember possible splitting points for each symbol.
# 
# *Hint:* You can call ``chart[start, end][x]`` to add symbol ``x`` to a cell but leave the set of splitting points empty (e.g. for level 1 at the bottom of the chart).
# 
# **Exercise (HARD):** Implement a ``get_trees`` method that uses the stored splitting points to compute all possible parse trees.
# 
# *Hints:*
# - If two trees have the same shape but different symbols in their inner nodes, they correspond to different parse trees.
# - You can represent a tree with a ``SetChart`` that only contains the symbols that appear in the tree.
# - Start from the top and construct trees recursively (for any possible split, any combination of a right sub-tree and a left sub-tree is a valid tree).

# In[10]:


class TreeCFG(CFG):
    def init_chart(self, n):
        # chart stores dicts from symbols to sets of possible splitting points
        self.chart = DictSetChart(n)
    
    def fill_terminal_cell(self, start, symbol):
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # FILL HERE
        for x, y in self.terminal_rules:
            if y == symbol:
                self.chart[start, start + 1][x]
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    def fill_non_terminal_cell(self, start, split, end):
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # FILL HERE
        for x1, x2, x3 in self.non_terminal_rules:
            if x2 in self.chart[start, split] and x3 in self.chart[split, end]:
                self.chart[start, end][x1].add((x2, x3, split))
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    def get_trees(self, symbol=None, start=None, end=None, chart=None, draw_tree=True):
        trees = []  # list of all possible trees represented in charts
        n = self.chart.n
        if (symbol, start, end, chart) == (None, None, None, None):
            (symbol, start, end, chart) = (self.start_symbol, 0, n, SetChart(n, empty="", left="", right=""))
        
        # handle terminal transitions
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # FILL HERE
        if end == start + 1:
            # make a copy and fill in current symbol
            chart_copy = chart.copy()
            chart_copy[start, end].add(symbol)
            trees += [chart_copy]
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        # go through all possible splitting points and the associated symbols
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # FILL HERE
        else:
            for left_child, right_child, split in self.chart[start, end][symbol]:
                # make a copy and fill in current symbol
                chart_copy = chart.copy()
                chart_copy[start, end].add(symbol)
                # fill in tree branches (only to make printing more pretty) 
                if draw_tree:
                    for tree_end in range(split + 1, end):
                        chart_copy[start, tree_end].add('╱')
                    for tree_start in range(start + 1, split):
                        chart_copy[tree_start, end].add('╲')
                # get sub-trees from left child (start from prefilled chart)
                left_trees = self.get_trees(symbol=left_child, start=start, end=split, chart=chart_copy, draw_tree=draw_tree)
                # for all possible left trees get sub-trees from right child
                for new_tree in left_trees:
                    # the left sub-tree is already filled in; now add the possible sub-trees from right child 
                    full_trees = self.get_trees(symbol=right_child, start=split, end=end, chart=new_tree, draw_tree=draw_tree)
                    # each combination is a valid (sub-)tree for the parent
                    trees += full_trees
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        return trees                


# Test your implementation again on the (slightly extended) minimal example.

# In[11]:


cfg = TreeCFG(
    non_terminal_rules=[
        ('A', 'A', 'A'),
        ('C', 'C', 'C'),
        ('A', 'C', 'C'),
        ('C', 'A', 'A'),
        ('A', 'B', 'A'),
        ('C', 'B', 'A')
    ],
    terminal_rules=[
        ('A', 'a'),
        ('B', 'b'),
    ],
    start_symbol='A'
)

print("valid sequence:", cfg.parse("aabba"))
print(cfg.chart.pretty())


# In[12]:


trees = cfg.get_trees()
print(f"There are {len(trees)} possible parse trees:")
print(cat_pretty(trees, crosses=True, grid_off=True))


# ## Working with Data

# ### Harmonic Annotations

# **[The Annotated Beethoven Corpus (ABC)](https://github.com/DCMLab/ABC)** is a dataset with expert harmonic analyses of all Beethoven string quartets. The harmonic labels contain more information and a broader variety of chords than what was possible to cover in the lectures. The following code loads data from one of the files (change ``file_idx`` to change the file) and extracts the basic numeral annotations. As you can see, these are still somewhat richer and more diverse than what we have covered.

# In[13]:


from muprocdurham import dcml_harmony_regex as regex
import corpusinterface as ci
import pandas

ci.reset_config('ABC_corpus.ini')

corpus = ci.load("ABC_harmonies_tsv", download=True)

file_name = 'n06op18-6_01.harmonies.tsv'
for idx, file in enumerate(corpus.files()):
    this_file_name = str(file).split("/")[-1]
    if this_file_name != file_name:
        continue
    print(this_file_name)
    data = pandas.read_csv(file, sep='\t')
    harmonies = list(data['label'])
    cleaned_harmonies = []
    for h in harmonies:
        match = regex.match(h)
        # print(f"Full: {h}, Chord: {match['chord']}, numeral: {match['numeral']}")  # uncomment for more info
        cleaned_harmonies.append(match['numeral'])
    print(cleaned_harmonies)


# **Exercise:** Use one of the following grammars (similar to those from the lecture slides) to parse the first couple of harmonies (try out different pieces) and check how many possible parse trees there are.

# In[14]:


major_grammar = TreeCFG(
    non_terminal_rules=[('I', 'V', 'I'),
                        ('IV', 'I', 'IV'),
                        ('vii0', 'IV', 'vii0'),
                        ('iii', 'vii0', 'iii'),
                        ('vi', 'iii', 'vi'),
                        ('ii', 'vi', 'ii'),
                        ('V', 'ii', 'V'),
                        ('I', 'I', 'I')],
    # relative
    terminal_rules=[('I', 'I'),
                    ('IV', 'IV'),
                    ('vii0', 'vii0'),
                    ('iii', 'iii'),
                    ('vi', 'vi'),
                    ('ii', 'ii'),
                    ('V', 'V')],
    # C Major
    # terminal_rules=[('I', 'C'),
    #                 ('IV', 'F'),
    #                 ('vii0', 'B0'),
    #                 ('iii', 'Em'),
    #                 ('vi', 'Am'),
    #                 ('ii', 'Dm'),
    #                 ('V', 'G')],
    start_symbol='I'
)

minor_grammar = TreeCFG(
    non_terminal_rules=[('i', 'v', 'i'),
                        ('i', 'V', 'i'),
                        ('iv', 'i', 'iv'),
                        ('VII', 'iv', 'VII'),
                        ('III', 'VII', 'III'),
                        ('VI', 'III', 'VI'),
                        ('ii0', 'VI', 'ii0'),
                        ('v', 'ii0', 'v'),
                        ('V', 'ii0', 'V'),
                        ('i', 'i', 'i')],
    # relative
    terminal_rules=[('i', 'i'),
                    ('iv', 'iv'),
                    ('VII', 'VII'),
                    ('III', 'III'),
                    ('VI', 'VI'),
                    ('ii0', 'ii0'),
                    ('V', 'V'),
                    ('v', 'v')],
    # A Minor
    # terminal_rules=[('i', 'Am'),
    #                 ('iv', 'Dm'),
    #                 ('VII', 'G'),
    #                 ('III', 'C'),
    #                 ('VI', 'F'),
    #                 ('ii0', 'B0'),
    #                 ('V', 'E'),
    #                 ('v', 'Em')],
    start_symbol='i'
)


# In[15]:


grammar = major_grammar
# grammar = minor_grammar
grammar.parse(cleaned_harmonies[:8])
print(grammar.chart.pretty())


# In[16]:


trees = grammar.get_trees()
print(f"{len(trees)} possible parse trees")
print(cat_pretty(trees, crosses=True, grid_off=True))
