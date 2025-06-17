### 并查集模板
字典式
```python
class DisjointSet:
    def __init__(self):
        self.parent = {}
        self.rank = {}
    def add(self, a):
        if a not in self.parent:
            self.parent[a] = a
            self.rank[a] = 0
    def find(self, x):
        self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        self.add(x)
        self.add(y)
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.rank[root_x] += 1
            self.parent[root_y] = root_x
        return True
    def connected(self, x, y):
        if x not in self.parent or y not in self.parent:
            return False
        return self.find(x) == self.find(y)
    
def slove(lines):
    d = DisjointSet()
    for line in lines:
        if line[1] == '=':
            d.union(line[0], line[3])
        if line[1] == '!':
            if d.connected(line[0], line[3]):
                return False
    return True

n = int(input())
lines = []
for _ in range(n):
    line = input()
    lines.append(line)
lines.sort(key=lambda x: ord(x[1]), reverse=True)

if slove(lines):
    print('True')
else:
    print('False')
```
一般式：
```python
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n - 1
        
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
            
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
                
            self.count -= 1
        
    def get_count(self):
        return self.count

case = 1
while True:
    try:
        n, m = map(int, input().strip().split())
        if n == 0 and m == 0:
            break
        ds = DisjointSet(n + 1)
        for i in range(m):
            a, b = map(int, input().strip().split())
            ds.union(a, b)
        ans = ds.get_count()
        print(f"Case {case}: {ans}")
        case += 1
    except EOFError:
        break
```

### Graph
判环的一种，亦可以使用拓扑排序
```python
color = [0] * n
def dfs(graph, color, node):
    if color[node] == 1:
        return True
    if color[node] == 2:
        return False
    
    color[node] = 1
    for neighbor in graph[node]:
        if dfs(graph, color, node):
            return True
    color[node] = 2
    return False
```

连通块的带权值类型，求连通块的权值和 or 连通块中的最大值
```python
def dfs(i, graph, cost, visited):
    visited.add(i)
    min_cost = cost[i - 1]
    for neighbor in graph[i]:
        if neighbor not in visited:
            min_cost = min(min_cost, dfs(neighbor, graph, cost, visited)) #可以改成 += 求total
    return min_cost
```

拓扑排序：
```python
def bfs(graph, s, n, dis):
    queue = deque([s])
    dis = [-1] * n
    dis[s] = 0
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if dis[neighbor] <= dis[node]:
                dis[neighbor] = dis[node] + 1
                queue.append(neighbor)
    return dis
```
无向图判环：记录上一个节点
```python
def is_cycle(graph, n):
    parent = [-1] * n
    visited = [0] * n
    def bfs(i):
        queue = deque([i])
        visited[i] = 1
        
        while queue:
            cur = queue.popleft()
            
            for neighbor in graph[cur]:
                if visited[neighbor] and parent[cur] != neighbor:
                    return True
                
                if not visited[neighbor]:
                    visited[neighbor] = 1
                    parent[neighbor] = cur
                    queue.append(neighbor)
                
        return False
```

### BFS
网络传送门，多一个传送的状态转移，由于传送不需要时间，故需要在搜到即入队，与搜到节点同层。
```python
while queue:
	x, y, dis = queue.popleft()
	if x == m - 1 and y == n - 1:
		return dis
	for dx, dy in directions:
		nx, ny = x + dx, y + dy
		if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] != '#' and dp[nx][ny] > dis + 1:
			dp[nx][ny] = dis + 1
			queue.append((nx, ny, dis + 1))

			if matrix[nx][ny] in gate:
				for i, j in gate[matrix[nx][ny]]:
					if not(i == x and j == y):
						queue.append((i, j, dis + 1))
				del gate[matrix[nx][ny]]
```
最大人工岛：
```python
from typing import List
# 对每个岛屿进行编号统计大小, 注意临界情况
class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        def dfs(x, y, visited, t, grid, size):
            visited[x][y] = t
            directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and visited[nx][ny] == 0 and grid[nx][ny] == 1:
                    size[t] += 1
                    dfs(nx, ny, visited, t, grid, size)
        t = 2
        size = {}
        visited = [[0] * n for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if grid[i][j] and visited[i][j] == 0:
                    size[t] = 1
                    dfs(i, j, visited, t, grid, size)
                    t += 1
        max_num = 0
        for i in range(m):
            for j in range(n):
                if not grid[i][j]:
                    cur = 1
                    island = {0, 1}
                    for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                        if 0 <= x < m and 0 <= y < n and visited[x][y] not in island:
                            cur += size[visited[x][y]]
                            island.add(visited[x][y])
                    max_num = max(max_num, cur)
        if max_num == 0:
            return m * n
        return max_num
```
调度场，单调栈，最小栈，子矩阵，数（并非二叉），MST，强连通分量，镜面映射，树上dp，哈夫曼编码数，字典树，词梯
### 拓扑排序，同时判环，唯一性和由几个等式来确定：
Sort it all out
```python
from collections import deque, defaultdict

def tp_sort(graph, n, indegree, tab):
    queue = deque()
    indegree = indegree.copy()
    for s in tab:
        if indegree[s] == 0:
            queue.append(s)
            
    multi = False
    result = []
    
    while queue:
        if len(queue) > 1:
            multi = True
        u = queue.popleft()
        result.append(u)
        for neighbor in graph[u]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
                
    if len(result) < n:
        return -1, []
    elif multi:
        return 0, []
    else:
        return 1, result
                
while True:
    n, m = map(int, input().split())
    if n == 0 and m == 0:
        break
    
    graph = defaultdict(list)
    indegree = defaultdict(int)
    tab = [chr(ord('A') + i) for i in range(n)]
    for s in tab:
        indegree[s] = 0
        
    find = False
    for i in range(m):
        line = input().strip()
        a, b = line[0], line[2]
        graph[a].append(b)
        indegree[b] += 1

        if find:
            continue
        
        k, result = tp_sort(graph, n, indegree, tab)
        if k == -1:
            print(f'Inconsistency found after {i + 1} relations.')
            find = True
        elif k == 1:
            ans = ''.join(result)
            print(f'Sorted sequence determined after {i + 1} relations: {ans}.')
            find = True

    if not find: # 最后才能确定是否可以被确定
        print("Sorted sequence cannot be determined.")
```
最小奖金方案：
```python
def tp_sort(graph, queue, dis, n):
    while queue:
        cur = queue.popleft()
        for neighbor in graph[cur]:
            if dis[neighbor] <= dis[cur]:
                dis[neighbor] = dis[cur] + 1
                queue.append(neighbor)
    
    extra = sum(dis[i] for i in range(n) if dis[i] != -1)
    return extra + n * 100

n, m = map(int, input().split())
graph = defaultdict(list)
queue = deque()
found = [False] * n
dis = [-1] * n
for _ in range(m):
    line = list(map(int, input().split()))
    graph[line[1]].append(line[0])
    found[line[0]] = True

for i in range(n):
    if not found[i]:
        dis[i] = 0
        queue.append(i)
        
print(tp_sort(graph, queue, dis, n))
```
### 动态规划DP
树dp：
```python
def rob(self, root: Optional[TreeNode]) -> int:
	def dfs(root):
		if root is None:
			return [0, 0]
		dp_left = dfs(root.left)
		dp_right = dfs(root.right)
		not_rob = max(dp_left) + max(dp_right)
		robbed = root.val + dp_left[0] + dp_right[0]
		return [not_rob, robbed]
	ans = dfs(root)
	return max(ans)
```
木材切割，枚举长度
```python
n = int(input())
m = int(input())
dp = [[0] * (n + 1) for _ in range(m + 1)]
price = list(map(int, input().strip().split()))
for i in range(1, m + 1):
    for j in range(i, n + 1):
        for k in range(1, j - i + 2): #确保每段长度大于1
            if i == 1:
                dp[i][j] = price[j - 1]
            else:
                dp[i][j] = max(dp[i][j], dp[i - 1][j - k] + price[k - 1])
            
print(dp[m][n])
```
最佳凑单：
```python
n, t = map(int, input().split())
prices = list(map(int, input().strip().split()))
dp = [float('inf')] * (t + max(prices) + 1)
dp[0] = 0
for i in range(n):
    for j in range(len(dp) - 1, prices[i] - 1, -1):
        dp[j] = min(dp[j], dp[j - prices[i]] + prices[i])
        
        
found = 0
for k in range(t, len(dp)):
    if dp[k] != float('inf'):
        found = 1
        print(dp[k])
        break
    
if not found:
    print(0)
```
单词拆分：
```python
from typing import List
# dp[i] 表示前 i 个字母可以被拆分
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        for i in range(n):
            for j in range(i + 1, n + 1):
                if dp[i] and s[i : j] in wordDict:
                    dp[j] = True

        return dp[n]
```
### 字符串解析问题，调度场算法：
dfs:
```python
def polish_notation(s):
    current = s.pop(0)
    if current in "+-*/":
        a = polish_notation(s)
        b = polish_notation(s)
        return eval(f"{a}{current}{b}")
    return float(current)
```

stack:
```python
def check(s):
    stack = []
    count = []  # 记录左括号的位置
    ans = [' '] * len(s)
    
    for i, char in enumerate(s):
        if char == '(':
            stack.append('(')
            count.append(i)
        elif char == ')':
            if not stack:
                ans[i] = '?'
            else:
                stack.pop()
                count.pop()
    
    for pos in count:
        ans[pos] = '$'
    
    print(''.join(s))
    print(''.join(ans))

while True:
    try:
        a = input().strip()
        s = list(a)
        check(s)
    except EOFError:
        break
```

```python
import sys
def decode(s, index):
    num = 0
    stack = []
    
    while index < len(s):
        if s[index] == '[':
            sub_str, next_index = decode(s, index + 1)
            stack.extend(sub_str)
            index = next_index
            
        elif s[index] == ']':
            return num * stack, index
        
        elif s[index].isdigit():  # 解析多位数
            num = num * 10 + int(s[index])
            
            ## 也可以 numstr = '' 然后逐位拼接 numstr += s[index]
            
        else:
            stack.append(s[index])
            
        index += 1
    
    return stack, index

s = sys.stdin.readline()
decode_str, _ = decode(s, 0)
print(*decode_str, sep = '')
```
回文dp：
```python
def min_insertions_to_palindrome(n, s):
    t = s[::-1]
    dp = [0] * (n + 1)

    for i in range(n):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if s[i] == t[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] =  max(dp[j], dp[j - 1]) #dp[j] 未更新时相当于 dp[i - 1][j] 更新后相当于 dp[i][j]
            prev = temp
    return n - dp[n]


n = int(input())
s = input().strip()
print(min_insertions_to_palindrome(n, s))
```
最长公共子序列：（滚动数组写法如上）
```python
# s1: 长度 m，s2: 长度 n
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    return dp[m][n]
```
最长公共子列：（可以维护索引）
```python
def longest_common_substring(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    res = 0

    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
                res = max(res, dp[i+1][j+1])
            # else dp[i+1][j+1] = 0  # 不写也行，初始值为 0
    return res
```
最长回文字串：(通过枚举字串来实现)
```python
def find_max_substr(str):
    n = len(str)
    if n < 2:
        return str
    max_len = 1
    dp = [[False] * n for _ in range(n)]
    
    for l in range(2, n + 1):
        for i in range(n):
            j = l + i - 1
            if j >= n:
                break
            if str[i] == str[j]:
                if l <= 3:
                    dp[i][j] = True
                    
                else:
                    dp[i][j] = dp[i + 1][j - 1]
                    
            if dp[i][j] and l > max_len:
                max_len = l
                begin = i
                
    return str[begin:begin + max_len]

s = input()
print(find_max_substr(s))
```
调度场算法：(中序表达式转后序)
```python
def transfer(s):
    
    priority = {'+': 1, '-': 1, '*': 2, '/': 2}
    char_stack = []
    ans_stack = []
    cur_num = ''
    
    for ch in s:
        if ch.isdigit() or ch == '.':
            cur_num += ch
        else:
            if cur_num:
                num = float(cur_num)
                ans_stack.append(int(num) if num.is_integer() else num)
                cur_num = ''
            """此处为两个if关系,旨在字符不再是数字时进行操作,但是不能影响其他字符的判断"""
            if ch in '+-*/':
                while char_stack and char_stack[-1] in '+-*/' and priority[ch] <= priority[char_stack[-1]]:
                    ans_stack.append(char_stack.pop())
                char_stack.append(ch)
            elif ch == '(':
                char_stack.append(ch)
            else:
                while char_stack and char_stack[-1] != '(':
                    ans_stack.append(char_stack.pop())
                char_stack.pop()
    """最后一位读取的如果是数字,其并未被压入stack"""
    if cur_num:
        num = float(cur_num)
        ans_stack.append(int(num) if num.is_integer() else num)
        
    while char_stack:
        ans_stack.append(char_stack.pop())
        
    return ' '.join(map(str, ans_stack))

n = int(input())
for _ in range(n):
    s = input().strip()
    ans = transfer(s)
    print(ans)
```
KMP算法(包括判断周期的部分)
```python
def kmp_lps(n, s):
    lps = [0] * n
    length = 0
    for i in range(1, n):
        while length > 0 and s[i] != s[length]:
            length = lps[length - 1]
        if s[length] == s[i]:
            length += 1
        lps[i] = length
    return lps
def match(text, sub):
    n = len(text)
    m = len(sub)
    if m == 0:
        return -1
    matches = []
    j = 0
    lps = kmp_lps(m, sub)
    for i in range(n):
        if j > 0 and text[i] != sub[j]:
            j = lps[j - 1]
        if text[i] == sub[j]:
            j += 1
        if j == m:
            matches.append((i - j + 1, i)) # text[i - j + 1:i + 1]
            j = lps[j - 1]
    return matches
cnt = 1
while True:
    n = int(input())
    if n == 0:
        break
    s = input().strip()
    next = kmp_lps(n, s)
    print(f'Test case #{cnt}')
    for i in range(2, n + 1):
        p = i - next[i - 1]
        if i % p == 0 and i // p > 1:
            print(i, i // p)
    print()
    cnt += 1
```
### Binary Search
toys：预处理转化，二分查找优化效率
明确所要查找的真正事物，去定义left，right
```python
def calculate_partition_x(U, y1, L, y2, toy_y):
    if y1 == y2:
        return U
    return U + (L - U) * (toy_y - y1) / (y2 - y1)
    
for _ in range(m):
	toy_x, toy_y = map(int, input().split())
	
	# Binary search to find the rightmost partition that toy_x is greater than
	left, right = 0, n - 1
	bin_num = 0
	
	while left <= right:
		mid = (left + right) // 2
		U, L = partitions[mid]
		partition_x = calculate_partition_x(U, y1, L, y2, toy_y)
		
		if toy_x > partition_x:
			bin_num = mid + 1  # toy is to the right of this partition
			left = mid + 1
		else:
			right = mid - 1
	
	count[bin_num] += 1

for i in range(n + 1):
	print(f'{i}: {count[i]}')
```
### Backtrack
棋盘问题：
```python
def backtrack(board, row, col, k, current, n, dp):
    if current == k:
        return 1
    if k - current > n - row:
        return 0
    count = 0
    # 选或不选
    for i in range(n):
        if i not in col and board[row][i] == '#':
            col.add(i)
            count += backtrack(board, row + 1, col, k, current + 1, n ,dp)
            col.remove(i)
    count += backtrack(board, row + 1, col, k, current, n, dp)
    return count
    print(backtrack(board, 0, col, k, 0, n, dp))
```
堆路径, 从右到左回溯
```python
def dfs(i, path, n, nums):
    path.append(nums[i - 1])
    left = 2 * i
    right = 2 * i + 1
    if left > n and right > n:
        paths.append(path[:])
    if right <= n:
        dfs(right, path, n, nums)
        path.pop()
    if left <= n:
        dfs(left, path, n, nums)
        path.pop()
is_max = all(path[i] >= path[i+1] for path in paths for i in range(len(path)-1))
is_min = all(path[i] <= path[i+1] for path in paths for i in range(len(path)-1))
```
出栈序列统计：
```python
def backtrack(pop_count, push_count, count, n):
    if pop_count == n:
        count[0] += 1
        return
    
    if push_count < n:
        stack.append(push_count + 1)
        backtrack(pop_count, push_count + 1, count, n)
        stack.pop()
        
    if stack:
        top = stack.pop()
        backtrack(pop_count + 1, push_count, count, n)
        stack.append(top)
        
n = int(input())
stack = []
count = [0]
backtrack(0, 0, count, n)
print(count[0])
```
N皇后排列图：
```python
from typing import List
# 主对角线上元素行列序号差 为定值
# 副对角线上元素行列序号和 为定值
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def generateBoard():
            board = []
            for i in range(n):
                row[queens[i]] = 'Q'
                board.append("".join(row))
                row[queens[i]] = '.'
            return board

        def backtrack(row, columns, dia1, dia2, ans):
            if row == n:
                board = generateBoard()
                ans.append(board)

            else:
                for i in range(n):
                    if i in columns or row - i in dia1 or row + i in dia2:
                        continue
                    queens[row] = i
                    columns.add(i)
                    dia1.add(row - i)
                    dia2.add(row + i)
                    backtrack(row + 1, columns, dia1, dia2, ans)
                    columns.remove(i)
                    dia1.remove(row - i)
                    dia2.remove(row + i)
        ans = []
        columns = set()
        dia1 = set()
        dia2 = set()
        queens = [-1] * n
        row = ['.'] * n
        backtrack(0, columns, dia1, dia2, ans)
        return ans
```
分割回文串：（枚举字串）：
```python
from typing import List
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)
        ans = []
        path = []

        def division(current_start):
            """
            考虑 s[current_start:]的分割
            """

            if current_start == n:
                ans.append(path[:])
                return
            
            for j in range(current_start, n):
                current = s[current_start: j + 1]
                if current == current[::-1]:
                    """
                    若j处为回文串, 可以分解, dfs(j + 1)
                    """
                    path.append(current)

                    division(j + 1)

                    path.pop()
        
        division(0)
        return ans
```
骑士周游：
```python
def is_valid(x, y, p, q):
    if x >= 0 and x < p and y >= 0 and y < q:
        return True
    return False

def back_track(x, y, p, q, deep, location, visited):
    if deep == p * q:
        return True
    
    
    directions = ((-1, -2), (1, -2), (-2, -1), (2, -1), (-2, 1), (2, 1), (-1, 2), (1, 2))
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if is_valid(nx, ny, p, q) and visited[nx][ny] == 0:
            visited[nx][ny] = 1
            location.append(chr(ny + ord('A')) + str(nx + 1))
            if back_track(nx, ny, p, q, deep + 1, location, visited):
                return True
            location.pop()
            visited[nx][ny] = 0
    return False
    
cnt = 1
n = int(input().strip())
for k in range(n):
    
    p, q = map(int, input().strip().split())
    find = False
    print(f"Scenario #{cnt}:")
    cnt += 1
    for j in range(q):
        if find:
            break
        for i in range(p):
            location = []
            location.append(chr(j + ord('A')) + str(i + 1))
            visited = [[0] * q for _ in range(p)]
            visited[i][j] = 1
            if back_track(i, j, p, q, 1, location, visited):
                print(''.join(location))
                print()
                find = True
                break
    if not find:
        print('impossible')
        print()
```
### 运算技巧集合上的运算：
倒排索引查询：
```python
def check(words, query, n):
    ans = set()
    first = True  # 用于标识是否是第一个 1，避免错误交集

    for i in range(n):
        if query[i] == 1:
            if first:
                ans = words[i + 1].copy()
                first = False
            else:
                ans &= words[i + 1]  # 取交集
    for i in range(n):
        if query[i] == -1:
            ans -= words[i + 1]  # 取差集
    if not ans:
        return "NOT FOUND"
    return " ".join(map(str, sorted(ans)))  # 按照升序输出

n = int(input())
words = defaultdict(set)

for i in range(1, n + 1):
    line = list(map(int, input().split()))
    ci = line[0]
    word_ids = line[1:ci + 1]
    words[i].update(word_ids)  # 迭代逐个输入 列表中为 extend

m = int(input())
for _ in range(m):
    query = list(map(int, input().split()))
    print(check(words, query, n))
```
取区间的交集：(雷达)
```python
import math
def min_radar_num(n, d, island):
    section = []
    for x, y in island:
        if y > d:
            return -1
        dx = math.sqrt(d * d - y * y)
        section.append((x - dx, x + dx))
        
    section.sort()
    
    num = 1
    last_end = section[0][1]
    
    for left, right in section:
        if left > last_end:
            num += 1
            last_end = right
        else:
            last_end = min(last_end, right)  # 更新右界很重要 update
    return num
```
自定义的defaultdict:
```python
def compute_icpc_rank(lines):
    all = defaultdict(lambda: {"solved": set(), "attempts": defaultdict(int), "total_attempts": 0})

    for line in lines:
        team, problem, result = line.split(',')
        team = team.strip()

        all[team]["attempts"][problem] += 1
        all[team]["total_attempts"] += 1

        if result.strip() == "yes":
            all[team]["solved"].add(problem)

    ranking = []
    for team, val in all.items():
        ranking.append((len(val["solved"]), val["total_attempts"], team))

    ranking.sort(key=lambda x: (-x[0], x[1], x[2]))
    
    for i, (solved, total_attempts, team) in enumerate(ranking[:12], 1):
        print(f'{i} {team} {solved} {total_attempts}') 
```
素数筛：
```python
def sieve(N):
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    
    for num in range(2, int(N**0.5) + 1):
        if is_prime[num]:
            for multiple in range(num * num, N + 1, num):
                is_prime[multiple] = False

    return is_prime

is_prime = sieve(10000)
```
### DFS
有限度的dfs：加一个depth限制
```python
def DLS(current, depth, visited, l, result):
    if depth > l:
        return
    result.append(current)
    visited[current] = True
       
    for neighbor in adj[current]:
        if not visited[neighbor]:
            DLS(neighbor, depth + 1, visited, l, result)
```
### MST
prim算法
```python
def prim(graph, n):
    heap = [(0, 0)]
    num_edges = 0
    visited = [False] * n
    ans = 0
    while num_edges < n:
        dis, i = heapq.heappop(heap)
        
        if visited[i]:
            continue
        
        visited[i] = True
        num_edges += 1
        ans += dis
        
        for neighbor, weight in graph[i]:
            heapq.heappush(heap, (weight, neighbor))
    return ans
ans = 0
while edge_num < n:
	dis, i = heapq.heappop(heap)
	if visited[i]:
		continue

	visited[i] = 1
	edge_num += 1
	ans += dis

	for j in range(n):
		if not visited[j]:
			dis = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
			heapq.heappush(heap, (dis, j))
return ans
```
### Stack
单调栈：
```python
def largestRectangleArea(self, heights: List[int]) -> int:
	n = len(heights)
	left = [0] * n
	right = [n] * n

	stack = []
	for i in range(n):
		while stack and heights[i] < heights[stack[-1]]:
			right[stack[-1]] = i
			stack.pop()
		left[i] = stack[-1] if stack else -1
		stack.append(i)
	ans = max((right[i] - left[i] - 1) * heights[i] for i in range(n))
	return ans
```
### Dijkstra
兔子与樱花：（名称与idx的映射）
```python
def Dijkstra(graph, start, end, name_to_idx, idx_to_name):
    n = len(name_to_idx)
    visited = [False] * n
    pre = [-1] * n
    dis = [float('inf')] * n
    dis[name_to_idx[start]] = 0
    heap = [(0, name_to_idx[start])]
    heapq.heapify(heap)
    while heap:
    ……
    path = []
    cur = name_to_idx[end]
    while cur != -1:
        path.append(cur)
        cur = pre[cur]
    path.reverse()
    
    result = idx_to_name[path[0]]
    for i in range(1, len(path)):
        u, v = path[i - 1], path[i]
        for k, cost in graph[u]:
            if k == v:
                result += f'->({cost})->{idx_to_name[k]}'
                break
    return result
P = int(input())
name_to_idx = {}
idx_to_name = {}
graph = defaultdict(list)
for i in range(P):
    name = input().strip()
    name_to_idx[name] = i
    idx_to_name[i] = name
Q = int(input())
for _ in range(Q):
    a, b, d = input().strip().split()
    d = int(d)
    u = name_to_idx[a]
    v = name_to_idx[b]
    graph[u].append((v, d))
    graph[v].append((u, d))
R = int(input())
for _ in range(R):
    start, end = input().split()
    print(Dijkstra(graph, start, end, name_to_idx, idx_to_name))
```
网络延迟时间：
```python
def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
	m = len(times)
	graph = defaultdict(list)
	for i in range(m):
		graph[times[i][0]].append((times[i][1], times[i][2]))
	
	dis = [float('inf')] * (n + 1)
	dis[0] = 0
	dis[k] = 0
	visited = [0] * (n + 1)
	heap = [(0, k)]
	while heap:
		time, cur = heapq.heappop(heap)

		if visited[cur]:
			continue

		visited[cur] = 1

		for neighbor, weight in graph[cur]:
			if dis[neighbor] > time + weight:
				dis[neighbor] = time + weight
				heapq.heappush(heap, (dis[neighbor], neighbor))
	ans = max(dis)
while heap:
        step, cur_x, cur_y = heapq.heappop(heap)
        
        if cur_x == ex and cur_y == ey:
            return step
        
        visited[cur_x][cur_y] = 1
        
        for dx, dy in directions:
            nx, ny = cur_x + dx, cur_y + dy
            if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] != '#' and not visited[nx][ny]:
                cost = abs(grid[nx][ny] - grid[cur_x][cur_y])
                if ans[nx][ny] > step + cost:
                    ans[nx][ny] = step + cost
                    heapq.heappush(heap, (ans[nx][ny], nx, ny))
                
    return None
```
有条件限制的Dijkstra，同样可以使用dp
```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        f = [float("inf")] * n
        f[src] = 0
        ans = float("inf")
        for t in range(1, k + 2):
            g = [float("inf")] * n
            for j, i, cost in flights:
                g[i] = min(g[i], f[j] + cost)
            f = g
            ans = min(ans, f[dst])
        
        return -1 if ans == float("inf") else ans
```
道路：
```python
def dijkstra(graph, n, K):
    heap = [(0, 0, 1)]
    distance = [float('inf')] * (n + 1)
    distance[1] = 0
    total_pay = [float('inf')] * (n + 1)
    total_pay[1] = 0
    while heap:
        dis, pay, cur = heapq.heappop(heap)
        
        if cur == n:
            return dis
        
        if dis >= distance[cur] and pay >= total_pay[cur] and not (dis == distance[cur] and pay == total_pay[cur]):
            continue
        
        for neighbor, length, cost in graph[cur]:
            new_dis = dis + length
            new_pay = pay + cost
            if new_pay <= K:
                heapq.heappush(heap, (new_dis, new_pay, neighbor))
                if new_dis < distance[neighbor]:
                    distance[neighbor] = new_dis
                    total_pay[neighbor] = new_pay
    return -1
```
### Bellman-Ford
```python
def get_profit(edges, n, s, v):
    dis = [0.0] * (n + 1) # 当前策略下 可持有 i 货币的最大数量
    dis[s] = v # 隐式确定以 s 为起点
    update = False
    for _ in range(n):
        update = False
        for a, b, rab, cab, rba, cba in edges:
            
            if dis[a] > cab:
                new_charge =  (dis[a] - cab) * rab
                if new_charge > dis[b]:
                    dis[b] = new_charge
                    update = True
            
            if dis[b] > cba:
                new_charge = (dis[b] - cba) * rba
                if new_charge > dis[a]:
                    dis[a] = new_charge
                    update = True                 
    if update == True:
        return 'YES'
    return 'NO'
```
### 差分数组：
```python
# 使用差分数组 记录开始与终止位置的状态 然后使用前缀和来表示 在该位置总操作次数
class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        pre = 0
        for l, r in queries:
            diff[l] += 1
            diff[r + 1] -= 1

        for i in range(n):
            pre += diff[i]
            if nums[i] - pre > 0:
                return False
        return True
import heapq
from typing import List
class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        prefix_sum = 0
        n = len(nums)
        diff = [0] * (n + 1)
        queries.sort()
        j = 0
        h = []
        for i in range(n):
            prefix_sum += diff[i]
            while j < len(queries) and queries[j][0] <= i:
                heapq.heappush(h, -queries[j][1])
                j += 1

            while h and -h[0] >= i and prefix_sum < nums[i]:
                diff[-h[0] + 1] -= 1
                heapq.heappop(h)
                prefix_sum += 1

            if prefix_sum < nums[i]:
                return -1
                
        return len(h)
```
### Tree
前缀树
```python
class Trie:
    def __init__(self):
        self.children = {}
        self.isend = False

    def insert(self, word: str) -> None:
        node = self
        for x in word:
            if x not in node.children:
                node.children[x] = Trie()
            node = node.children[x]
        node.isend = True
        
    def search(self, word: str) -> bool:
        cur = self
        for x in word:
            if x not in cur.children:
                return False
            cur = cur.children[x]

        return cur.isend

    def startsWith(self, prefix: str) -> bool:
        cur = self
        for x in prefix:
            if x not in cur.children:
                return False
            cur = cur.children[x]
        return True
```
树的遍历：
```python
def preorder_traversal(root):
    """前序遍历：根 -> 左 -> 右"""
    return root.value + preorder_traversal(root.left) + preorder_traversal(root.right) if root else ""

def inorder_traversal(root):
    """中序遍历：左 -> 根 -> 右"""
    return inorder_traversal(root.left) + root.value + inorder_traversal(root.right) if root else ""
# 前序遍历
def preorder_traversal(node):
    if node:
        print(node.value, end=" ")
        preorder_traversal(node.left)
        preorder_traversal(node.right)

def inorder(node, res):
    if node is None:
        return
    inorder(node.left, res)
    res.append(node.val)
    inorder(node.right, res)

def postorder(node, res):
    if node is None:
        return
    postorder(node.left, res)
    postorder(node.right, res)
    res.append(node.val)
```
叶子与高度：
```python
def tree_height(node):
    if node is None:
        return -1  # 根据定义，空树高度为-1
    return max(tree_height(node.left), tree_height(node.right)) + 1

def count_leaves(node):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)
has_parent = [False] * (n + 1)
```
两棵树是否相同：
```python
def is_same_tree(p, q):
    if not p and not q:
        return True
    if not p or not q:
        return False
    return (p.val == q.val and
            is_same_tree(p.left, q.left) and
            is_same_tree(p.right, q.right))
```
翻转二叉树：
```python
def invert_tree(root):
    if root:
        root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root
```
括号嵌套：
```python
def parse_tree(s):
    """ 解析括号嵌套格式的二叉树 """
    if s == '*':  # 处理空树
        return None
    if '(' not in s:  # 只有单个根节点
        return TreeNode(s)

    root_value = s[0]  # 根节点值
    subtrees = s[2:-1]  # 去掉根节点和外层括号

    # 使用栈找到逗号位置
    stack = []
    comma_index = None
    for i, char in enumerate(subtrees):
        if char == '(':
            stack.append(char)
        elif char == ')':
            stack.pop()
        elif char == ',' and not stack:
            comma_index = i
            break

    left_subtree = subtrees[:comma_index] if comma_index is not None else subtrees
    right_subtree = subtrees[comma_index + 1:] if comma_index is not None else None

    root = TreeNode(root_value)
    root.left = parse_tree(left_subtree)  # 解析左子树
    root.right = parse_tree(right_subtree) if right_subtree else None  # 解析右子树
    return root
```
扩展先序建树：
```python
def build_tree(s, index):
    # 如果当前字符为'.'，表示空结点，返回None，并将索引后移一位
    if s[index] == '.':
        return None, index + 1
    # 否则创建一个结点
    node = Node(s[index])
    index += 1
    # 递归构造左子树
    node.left, index = build_tree(s, index)
    # 递归构造右子树
    node.right, index = build_tree(s, index)
    return node, index
```
建立二叉搜索树：
```python
def insert(root, val):
    if root is None:
        return TreeNode(val)
    if val < root.val:
        root.left = insert(root.left, val)
    else:
        root.right = insert(root.right, val)
    return root
def build_bst(arr):
    root = None
    for val in arr:
        root = insert(root, val)
    return root
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        res = []
        def inorder(node):
            if not node or len(res) == k:
                return 
            inorder(node.left)
            res.append(node.val)
            inorder(node.right)
        inorder(root)
        return res[k - 1]
```
括号嵌套：
```python
def parse_tree(s):
    stack = []
    node = None
    for char in s:
        if char.isalpha():  # 如果是字母，创建新节点
            node = TreeNode(char)
            if stack:  # 如果栈不为空，把节点作为子节点加入到栈顶节点的子节点列表中
                stack[-1].children.append(node)
        elif char == '(':  # 遇到左括号，当前节点可能会有子节点
            if node:
                stack.append(node)  # 把当前节点推入栈中
                node = None
        elif char == ')':  # 遇到右括号，子节点列表结束
            if stack:
                node = stack.pop()  # 弹出当前节点
    return node  # 根节点
def preorder(node):
    output = [node.value]
    for child in node.children:
        output.extend(preorder(child))
    return ''.join(output)

def postorder(node):
    output = []
    for child in node.children:
        output.extend(postorder(child))
    output.append(node.value)
    return ''.join(output)```
按照大小遍历树：
```python
for _ in range(n):
        parts = list(map(int, sys.stdin.readline().split()))
        parent, *children = parts
        tree[parent].extend(children)
        all_nodes.add(parent)
        all_nodes.update(children)
        child_nodes.update(children)
    
    # 根节点 = 出现在 all_nodes 但没出现在 child_nodes 的那个
    root = (all_nodes - child_nodes).pop()
    
    def traverse(u):
        # 把 u 自己和它的所有直接孩子放一起排序
        group = tree[u] + [u]
        group.sort()
        for x in group:
            if x == u:
                print(u)
            else:
                traverse(x)
    
    traverse(root)
```
层序遍历树：
```python
queue = deque([root])  # Initialize the queue with the root node

    while queue:
        level_size = len(queue)  # Number of nodes at the current level
        # Process all nodes in the current level
        for _ in range(level_size):
            current_node = queue.popleft()  # Dequeue the front node
            print(current_node.key, end=' ')  # Print the node's key

            # Enqueue all children of the current node
            for child in current_node.children:
                queue.append(child)
```
前序与中序遍历建树：
```python
def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root_value = preorder[0]
    root = TreeNode(root_value)
    root_index_inorder = inorder.index(root_value)
    root.left = build_tree(preorder[1:1+root_index_inorder], inorder[:root_index_inorder])
    root.right = build_tree(preorder[1+root_index_inorder:], inorder[root_index_inorder+1:])
    return root
```
中后序遍历建树：
```python
def buildTree(inorder, postorder):
    if not inorder or not postorder:
        return None

    # 后序遍历的最后一个元素是当前的根节点
    root_val = postorder.pop()
    root = TreeNode(root_val)

    # 在中序遍历中找到根节点的位置
    root_index = inorder.index(root_val)

    # 构建右子树和左子树
    root.right = buildTree(inorder[root_index + 1:], postorder)
    root.left = buildTree(inorder[:root_index], postorder)

    return root
```
