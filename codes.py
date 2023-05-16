laCode = """file = open("add.c", 'r') lines = file.readlines()
keywords = ["void", "main", "int", "float", "bool", "if", "for", "else", "while", "char", "return"] operators = ["=", "==", "+", "-", "*", "/", "++", "--", "+=", "-=", "!=", "||", "&&"] punctuations= [";", "(", ")", "{", "}", "[", "]"]
def is_int(x): try:
EX. NO. 1
int(x)
return True except:
return False
for line in lines:
for i in line.strip().split(" "):
if i in keywords:
print (i, " is a keyword")
elif i in operators:
print (i, " is an operator")
elif i in punctuations:
print (i, " is a punctuation")
elif is_int(i):
print (i, " is a number")
else:
print (i, " is an identifier")
""".strip()

reNFACode = """
transition_table = [ [0]*3 for _ in range(20) ]
re = input("Enter the regular expression : ") re += " "
i =0
j =1 while(i<len(re)):
if re[i] == 'a': try:
if re[i+1] != '|' and re[i+1] !='*':
 transition_table[j][0] = j+1
j += 1
elif re[i+1] == '|' and re[i+2] =='b':
transition_table[j][2]=((j+1)*10)+(j+3) j+=1
transition_table[j][0]=j+1
j+=1
transition_table[j][2]=j+3 j+=1 transition_table[j][1]=j+1 j+=1 transition_table[j][2]=j+1 j+=1
i=i+2
elif re[i+1]=='*':
transition_table[j][2]=((j+1)*10)+(j+3) j+=1
transition_table[j][0]=j+1
j+=1 transition_table[j][2]=((j+1)*10)+(j-1) j+=1
except:
transition_table[j][0] = j+1
elif re[i] == 'b': try:
if re[i+1] != '|' and re[i+1] !='*': transition_table[j][1] = j+1
j += 1
elif re[i+1]=='|' and re[i+2]=='a': transition_table[j][2]=((j+1)*10)+(j+3)

 j+=1 transition_table[j][1]=j+1 j+=1 transition_table[j][2]=j+3 j+=1 transition_table[j][0]=j+1 j+=1 transition_table[j][2]=j+1 j+=1
i=i+2
elif re[i+1]=='*': transition_table[j][2]=((j+1)*10)+(j+3) j+=1
transition_table[j][1]=j+1
j+=1 transition_table[j][2]=((j+1)*10)+(j-1) j+=1
except:
transition_table[j][1] = j+1
elif re[i]=='e' and re[i+1]!='|'and re[i+1]!='*': transition_table[j][2]=j+1
j+=1
elif re[i]==')' and re[i+1]=='*':
transition_table[0][2]=((j+1)*10)+1 transition_table[j][2]=((j+1)*10)+1 j+=1
i +=1

print ("Transition function:") for i in range(j):
if(transition_table[i][0]!=0): print("q[{0},a]-->{1}".format(i,transition_table[i][0]))
if(transition_table[i][1]!=0): print("q[{0},b]-->{1}".format(i,transition_table[i][1]))
if(transition_table[i][2]!=0): if(transition_table[i][2]<10):
print("q[{0},e]-->{1}".format(i,transition_table[i][2])) else:
print("q[{0},e]-->{1} & {2}".format(i,int(transition_table[i][2]/10),transition_table[i][2]%10))
""".strip()


ndaDFACode = """
import pandas as pd
nfa = {}
n = int(input("No. of states : "))
t = int(input("No. of transitions : ")) for i in range(n):
state = input("state name : ") nfa[state] = {}
for j in range(t):
path = input("path : ")
EX. NO. 3
 print("Enter end state from state {} travelling through path {} : ".format(state, path)) reaching_state = [x for x in input().split()]
nfa[state][path] = reaching_state
print("\nNFA :- \n")
print(nfa)
print("\nPrinting NFA table :- ") nfa_table = pd.DataFrame(nfa) print(nfa_table.transpose())
print("Enter final state of NFA : ") nfa_final_state = [x for x in input().split()]
new_states_list = [] dfa = {}
keys_list = list(
list(nfa.keys())[0])
path_list = list(nfa[keys_list[0]].keys())
dfa[keys_list[0]] = {} for y in range(t):
var = "".join(nfa[keys_list[0]][ path_list[y]])
dfa[keys_list[0]][path_list[y]] = var if var not in keys_list:
new_states_list.append(var) keys_list.append(var)
while len(new_states_list) != 0: dfa[new_states_list[0]] = {}
for _ in range(len(new_states_list[0])):

for i in range(len(path_list)): temp = []
for j in range(len(new_states_list[0])):
temp += nfa[new_states_list[0][j]][path_list[i]]
s = ""
s = s.join(temp)
if s not in keys_list:
new_states_list.append(s)
keys_list.append(s) dfa[new_states_list[0]][path_list[i]] = s
new_states_list.remove(new_states_list[0])
print("\nDFA :- \n")
print(dfa)
print("\nPrinting DFA table :- ") dfa_table = pd.DataFrame(dfa) print(dfa_table.transpose())
dfa_states_list = list(dfa.keys()) dfa_final_states = []
for x in dfa_states_list:
for i in x:
if i in nfa_final_state:
dfa_final_states.append(x) break
print("\nFinal states of the DFA are : ", dfa_final_states)
""".strip()


leftRCode = """
def eliminate_left_recursion(grammar):
    non_terminals = list(grammar.keys())

    for i in range(len(non_terminals)):
        for j in range(i):
            for k in range(len(grammar[non_terminals[i]])):
                if grammar[non_terminals[i]][k].startswith(non_terminals[j]):
                    new_productions = []
                    for p in grammar[non_terminals[i]]:
                        if p.startswith(non_terminals[j]):
                            for q in grammar[non_terminals[j]]:
                                new_productions.append(q + p[len(non_terminals[j]):])
                        else:
                            new_productions.append(p)
                    grammar[non_terminals[i]] = new_productions

    return grammar


# Example usage
grammar = {
    'S': ['SAB', 'B'],
    'A': ['AC', 'a'],
    'B': ['BB', 'b'],
    'C': ['c']
}

new_grammar = eliminate_left_recursion(grammar)
for non_terminal, productions in new_grammar.items():
    print(non_terminal + ' -> ' + ' | '.join(productions))
"""


leftFCode = """
def left_factoring(grammar):
    non_terminals = list(grammar.keys())

    for non_terminal in non_terminals:
        productions = grammar[non_terminal]
        common_prefixes = find_common_prefixes(productions)

        if common_prefixes:
            new_productions = []
            for prefix in common_prefixes:
                new_non_terminal = non_terminal + "'"
                new_productions.append(prefix + new_non_terminal)
                new_productions.extend(
                    [production[len(prefix):] or 'ε' for production in productions if production.startswith(prefix)]
                )

            grammar[non_terminal] = new_productions
            grammar[new_non_terminal] = [production[len(prefix):] or 'ε' for production in productions if
                                          not production.startswith(prefix)]

    return grammar


def find_common_prefixes(productions):
    common_prefixes = []

    for i in range(len(productions[0])):
        prefixes = set([production[:i + 1] for production in productions])
        if len(prefixes) == 1:
            common_prefixes.append(productions[0][:i + 1])
        else:
            break

    return common_prefixes


# Example usage
grammar = {
    'S': ['abc', 'abd', 'efg', 'efh'],
    'A': ['xy', 'xz', 'xa'],
    'B': ['pqrs', 'pqrt', 'uvw']
}

new_grammar = left_factoring(grammar)
for non_terminal, productions in new_grammar.items():
    print(non_terminal + ' -> ' + ' | '.join(productions))
"""


firstFollowCode = """
import re
import string import pandas as pd
def parse(user_input,start_symbol,parsingTable):
#flag flag = 0
#appending dollar to end of input user_input = user_input + "$"
stack = []
stack.append("$") stack.append(start_symbol)
input_len = len(user_input) index = 0
while len(stack) > 0:
#element at top of stack top = stack[len(stack)-1]
print ("Top =>",top) #current input
 current_input = user_input[index]
print ("Current_Input => ",current_input)
if top == current_input: stack.pop()
index = index + 1 else:
#finding value for key in table key = top , current_input
print (key)
#top of stack terminal => not accepted if key not in parsingTable:
flag = 1 break
value = parsingTable[key] if value !='@':
value = value[::-1] value = list(value)
#poping top of stack stack.pop()
#push value chars to stack for element in value:
stack.append(element) stack.pop()
else:

 if flag == 0:
print ("String accepted!")
else:
print ("String not accepted!")
def ll1(follow, productions):
print ("\nParsing Table\n")
table = {}
for key in productions:
for value in productions[key]: if value!='@':
for element in first(value, productions): table[key, element] = value
else:
for element in follow[key]:
for key,val in table.items(): print (key,"=>",val)
new_table = {} for pair in table:
new_table[pair[1]] = {}
for pair in table:
new_table[pair[1]][pair[0]] = table[pair]
table[key, element] = value

 print ("\n")
print ("\nParsing Table in matrix form\n") print (pd.DataFrame(new_table).fillna('-')) print ("\n")
return table
def follow(s, productions, ans): if len(s)!=1 :
return {}
for key in productions:
for value in productions[key]:
f = value.find(s) if f!=-1:
if f==(len(value)-1): if key!=s:
if key in ans:
temp = ans[key]
else:
ans = follow(key, productions, ans) temp = ans[key]
ans[s] = ans[s].union(temp)
else:
first_of_next = first(value[f+1:], productions) if '@' in first_of_next:
if key!=s:
if key in ans:
temp = ans[key]

 return ans
def first(s, productions): c = s[0]
ans = set()
if c.isupper():
else:
ans = follow(key, productions, ans) temp = ans[key]
ans[s] = ans[s].union(temp)
ans[s] = ans[s].union(first_of_next) - {'@'}
else:
ans[s] = ans[s].union(first_of_next)
for st in productions[c]: if st == '@' :
if len(s)!=1 :
ans = ans.union( first(s[1:], productions) )
else :
ans = ans.union('@')
else :
f = first(st, productions)
ans = ans.union(x for x in f)
else:
ans = ans.union(c)
return ans
if __name__=="__main__": productions=dict()
grammar = open("grammar2", "r") first_dict = dict()
follow_dict = dict()

 flag = 1
start = ""
for line in grammar:
l = re.split("( |->|\n|\||)*", line) lhs = l[0]
rhs = set(l[1:-1])-{''}
if flag :
flag = 0
start = lhs productions[lhs] = rhs
print ('\nFirst\n')
for lhs in productions:
first_dict[lhs] = first(lhs, productions) for f in first_dict:
print (str(f) + " : " + str(first_dict[f])) print ("")
print ('\nFollow\n')
for lhs in productions: follow_dict[lhs] = set()
follow_dict[start] = follow_dict[start].union('$')
for lhs in productions:
follow_dict = follow(lhs, productions, follow_dict)
for lhs in productions:
follow_dict = follow(lhs, productions, follow_dict)

for f in follow_dict:
print (str(f) + " : " + str(follow_dict[f]))
ll1Table = ll1(follow_dict, productions)
#parse("a*(a+a)",start,ll1Table) parse("ba=a+23",start,ll1Table)
# tp(ll1Table)

"""


postpCODE = """
def infix_to_postfix(expression):
    operators = {
        '+': 1,
        '-': 1,
        '*': 2,
        '/': 2,
        '^': 3
    }
    stack = []
    postfix = ''

    for char in expression:
        if char.isalnum():
            postfix += char
        elif char == '(':
            stack.append('(')
        elif char == ')':
            while stack and stack[-1] != '(':
                postfix += stack.pop()
            stack.pop()
        else:
            while stack and stack[-1] != '(' and operators[char] <= operators.get(stack[-1], 0):
                postfix += stack.pop()
            stack.append(char)

    while stack:
        postfix += stack.pop()

    return postfix


def infix_to_prefix(expression):
    operators = {
        '+': 1,
        '-': 1,
        '*': 2,
        '/': 2,
        '^': 3
    }
    stack = []
    prefix = ''

    for char in expression[::-1]:
        if char.isalnum():
            prefix = char + prefix
        elif char == ')':
            stack.append(')')
        elif char == '(':
            while stack and stack[-1] != ')':
                prefix = stack.pop() + prefix
            stack.pop()
        else:
            while stack and stack[-1] != ')' and operators[char] <= operators.get(stack[-1], 0):
                prefix = stack.pop() + prefix
            stack.append(char)

    while stack:
        prefix = stack.pop() + prefix

    return prefix


# Example usage
infix_expression = 'A + B * C - D / E ^ F'
postfix_expression = infix_to_postfix(infix_expression)
prefix_expression = infix_to_prefix(infix_expression)

print("Infix Expression:", infix_expression)
print("Postfix Expression:", postfix_expression)
print("Prefix Expression:", prefix_expression)
"""


dagCode = """
class DAGNode:
    def __init__(self, label, operator=None, operand1=None, operand2=None):
        self.label = label
        self.operator = operator
        self.operand1 = operand1
        self.operand2 = operand2
        self.parents = []


def construct_dag(expression):
    symbol_table = {}
    dag_nodes = []

    def create_dag_node(label, operator=None, operand1=None, operand2=None):
        node = DAGNode(label, operator, operand1, operand2)
        dag_nodes.append(node)
        return node

    def get_dag_node(label):
        return symbol_table.get(label)

    def add_dag_parent(node, parent):
        node.parents.append(parent)

    def evaluate_expression(expr):
        if expr.isdigit():
            return create_dag_node(expr)

        tokens = expr.split()
        operator = tokens[0]
        operand1 = get_dag_node(tokens[1])
        operand2 = get_dag_node(tokens[2])

        if operand1 is None:
            operand1 = create_dag_node(tokens[1])

        if operand2 is None:
            operand2 = create_dag_node(tokens[2])

        result = operator + str(len(dag_nodes) + 1)
        node = create_dag_node(result, operator, operand1, operand2)

        add_dag_parent(operand1, node)
        add_dag_parent(operand2, node)

        symbol_table[result] = node

        return node

    root_node = evaluate_expression(expression)
    return root_node, dag_nodes


# Example usage
expression = "a + b * (c - d)"
root, dag_nodes = construct_dag(expression)

print("DAG Nodes:")
for node in dag_nodes:
    parents = [parent.label for parent in node.parents]
    print(f"Label: {node.label}, Operator: {node.operator}, Operand1: {node.operand1}, Operand2: {node.operand2}, Parents: {parents}")

print("\nRoot Node:")
print(f"Label: {root.label}, Operator: {root.operator}, Operand1: {root.operand1}, Operand2: {root.operand2}, Parents: {root.parents}")
"""

leadTrailCode = """
def compute_leading_sets(grammar):
    leading = {}
    epsilon = 'ε'

    # Initialize leading sets with empty sets
    for non_terminal in grammar['non_terminals']:
        leading[non_terminal] = set()

    # Iterate until no changes occur in the leading sets
    changed = True
    while changed:
        changed = False

        for production in grammar['productions']:
            non_terminal = production[0]
            rhs = production[1:]

            for symbol in rhs:
                if symbol in grammar['terminals']:
                    # Add terminal symbol to the leading set
                    if symbol not in leading[non_terminal]:
                        leading[non_terminal].add(symbol)
                        changed = True

                    break
                else:
                    # Add leading set of the non-terminal symbol to the current non-terminal's leading set
                    if epsilon not in leading[symbol]:
                        leading[non_terminal] = leading[non_terminal].union(leading[symbol])
                        changed = True

                    if epsilon not in leading[symbol]:
                        break

    return leading


def compute_trailing_sets(grammar):
    trailing = {}
    epsilon = 'ε'

    # Initialize trailing sets with empty sets
    for non_terminal in grammar['non_terminals']:
        trailing[non_terminal] = set()

    # Iterate until no changes occur in the trailing sets
    changed = True
    while changed:
        changed = False

        for production in grammar['productions']:
            non_terminal = production[0]
            rhs = production[1:]
            n = len(rhs)

            for i in range(n - 1, -1, -1):
                symbol = rhs[i]

                if symbol in grammar['terminals']:
                    # Add terminal symbol to the trailing set
                    if symbol not in trailing[non_terminal]:
                        trailing[non_terminal].add(symbol)
                        changed = True

                    break
                else:
                    # Add trailing set of the non-terminal symbol to the current non-terminal's trailing set
                    if epsilon not in trailing[symbol]:
                        trailing[non_terminal] = trailing[non_terminal].union(trailing[symbol])
                        changed = True

                    if epsilon not in trailing[symbol]:
                        break

    return trailing


# Example usage
grammar = {
    'non_terminals': ['E', 'T', 'F'],
    'terminals': ['+', '*', '(', ')', 'id'],
    'productions': [
        ['E', 'T', '+', 'E'],
        ['E', 'T'],
        ['T', 'F', '*', 'T'],
        ['T', 'F'],
        ['F', '(', 'E', ')'],
        ['F', 'id']
    ]
}

leading_sets = compute_leading_sets(grammar)
trailing_sets = compute_trailing_sets(grammar)

print("Leading Sets:")
for non_terminal, leading_set in leading_sets.items():
    print(f"{non_terminal}: {leading_set}")

print("\nTrailing Sets:")
for non_terminal, trailing_set in trailing_sets.items():
    print(f"{non_terminal}: {trailing_set}")

"""
