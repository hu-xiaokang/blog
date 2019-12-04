---
layout: post
title:  "Algorithms"
date:  2019-12-04 16:48:00
img: 5/main.jpg
description: 一些常用的算法
---

+ Partition函数

  ```java
  /**
  * 对[left,right]范围数据进行partition操作
  **/
  public int partition(int[] nums,int left,int right) {
      //随机选择一个中间值下标（为实现简单也可以直接选择right）
      int midIdx = getRandom(left,right);
      int mid = nums[midIdx];
      //把midIdx交换到right
      swap(nums,midIdx,right);
      //把小于mid的数交换到smallIdx位置
      int smallIdx = left - 1;
      for(int i=left;i<right;i++) {
          if(nums[i] < mid) {
              smallIdx++;
              swap(nums,smallIdx,i);
          }
      }
      smallIdx++;
      swap(nums,smallIdx,right);
      return smallIdx;
  }
  /**
  * 从[left,right]随机获取一个值
  **/
  private int getRandom(int left,int right) {
      int idx = left + (int)((right-left+1)*Math.random());
      return idx;
  }
  /**
  * 交换nums数组中下标i,j的值
  **/
  private void swap(int[] nums,int i,int j) {
      if(i != j) {
          int tem = nums[i];
          nums[i] = nums[j];
          nums[j] = tem;
      }
  }
  ```

  实现快速排序（时间复杂度o(nlog(n))）

  ```java
  /**
  * https://leetcode.com/problems/sort-an-array/submissions/
  **/
  public List<Integer> sortArray(int[] nums) {
      quickSort(nums);
      List<Integer> list = new LinkedList<>();
      for(int num : nums) {
          list.add(num);
      }
      return list;
  }
  /**
  * 进行快速排序
  **/
  public void quickSort(int[] nums) {
      if(nums != null&&nums.length > 1) {
          quickSort0(nums, 0, nums.length-1);
      }
  }
  /**
  * 快速排序
  **/
  private void quickSort0(int[] nums, int left, int right) {
      if(left < right) {
          int midIdx = partition(nums, left, right);
          quickSort0(nums,left,midIdx-1);
          quickSort0(nums, midIdx+1, right);
      }
  }
  ```

  解决TopK问题（时间复杂度o(n)）（**寻找中位数也是一个TopK问题**）

  ```java
  /**
  * 从nums中选出第k大的数（这种方法会修改原始输入数组）
  * https://leetcode.com/problems/kth-largest-element-in-an-array/
  **/
   public int findKthLargest(int[] nums, int k) {
      //非法输入处理
      if(nums == null || k <= 0 || k > nums.length) {
          return -1;
      }
      int left = 0;
      int right = nums.length - 1;
      //第k大数对应的序号
       int seq = nums.length - k;
      //二分查找
      while(left <= right) {
          int midIdx = partition(nums, left, right);
          if(midIdx < seq) {
              left = midIdx + 1;
          } else if(midIdx > seq) {
              right = midIdx-1;
          } else {
              return nums[midIdx];
          }
      }
      return nums[left];
  }
  ```

+ 回溯法

  **基本模板**

  ```java
  public void function() {
      //进行递归搜索
      backtracking();
  }
  private void backtracking() {
      //判断是否找到解(重要：递归结束条件)
      if () {
          //对解进行处理，然后返回
          return;
      }
       //进行剪枝
      if() {
          //对不满足解进行剪枝
          return;
      }
      //对解空间进行搜索
      for () {
          //在该点上的一些操作
          doSomethingBefore();
          //进入下一个点进行搜索
          backtracking();
          //从该点返回的一些操作
          doSomethingAfter();
      }
  }
  ```

  路径搜索

  ```java
  /**
  * https://leetcode.com/problems/longest-increasing-path-in-a-matrix/
  * 方式一：暴力遍历（从每个节点进行路径搜索，超时了）
  **/
  public int longestIncreasingPath(int[][] matrix) {
      if(matrix==null||matrix.length<1) {
          return 0;
      }
      maxPath = 1;
      //遍历所有点，并从该点进行搜索
      for(int row=0;row<matrix.length;row++) {
          for(int col=0;col<matrix[0].length;col++) {
              backtracking(matrix,row,col,1);
          }
      }
      return maxPath;
  }
  //表示移动的四个方向
  private int[][] directs = { {0,1}, {0,-1}, {1,0}, {-1,0} };
  private int maxPath = 1;
  private void backtracking(int[][] matrix, int curRow,int curCol,
                           int curLength) {
      if(curLength>maxPath) {
          maxPath = curLength;
      }
      //从当前点往上下左右进行移动
      for(int[] direct : directs) {
          int nextRow = curRow  + direct[1];
          int nextCol = curCol + direct[0];
          if(nextRow>=0&&nextRow<matrix.length
             &&nextCol>=0&&nextCol<matrix[0].length
             &&matrix[nextRow][nextCol]>matrix[curRow][curCol]) {
              backtracking(matrix,nextRow,nextCol,curLength+1);
          }
      }
  }
  ```

  ```java
  /**
  * https://leetcode.com/problems/longest-increasing-path-in-a-matrix/
  * 方式二：使用备忘录
  **/
  public int longestIncreasingPath(int[][] matrix) {
      if(matrix==null||matrix.length<1) {
          return 0;
      }
      int maxPath = 1;
      //记录从当前点搜索的最大路径长度
      int[][] memo = new int[matrix.length][matrix[0].length];
      //遍历所有点，并从该点进行搜索
      for(int row=0;row<matrix.length;row++) {
          for(int col=0;col<matrix[0].length;col++) {
             int len = backtracking(matrix,memo,row,col);
              maxPath = Math.max(len, maxPath);
          }
      }
      return maxPath;
  }
  //表示移动的四个方向
  private int[][] directs = {{0,1},{0,-1},{1,0},{-1,0}};
  private int backtracking(int[][] matrix,int[][] memo, 
                            int curRow,int curCol) {
      //该点的最大搜索路径长度已记录过
      if( memo[curRow][curCol] != 0) {
          return memo[curRow][curCol];
      }
      int pathLen = 1;
      //从当前点往上下左右进行移动
      for(int[] direct : directs) {
          int nextRow = curRow  + direct[1];
          int nextCol = curCol + direct[0];
          if(nextRow>=0&&nextRow<matrix.length
             &&nextCol>=0&&nextCol<matrix[0].length
             &&matrix[nextRow][nextCol]>matrix[curRow][curCol]) {
              int len = backtracking(matrix,memo,nextRow,nextCol)+1;
              pathLen = Math.max(pathLen, len);
          }
      }
      //记录该点的最长路径长度
      memo[curRow][curCol] = pathLen;
      return pathLen;
  }
  ```

  排列组合（基本思想就是对所有解进行遍历，并及时对不满足的解进行剪枝操作）

  ```java
  /**
  * https://leetcode.com/problems/permutations/
  * 方式一：利用一个boolean数组，记录已访问过的点
  **/
  public List<List<Integer>> permute(int[] nums) {
      List<List<Integer>> result = new LinkedList<>();
      List<Integer> list = new LinkedList<>();
      //用来记录已访问过的点
      boolean[] visited = new boolean[nums.length];
       //对解空间进行搜索
      backtracking(nums, visited, result, list);
      return result;
  }
  private void backtracking(int[] nums, boolean[] visited,
                    List<List<Integer>> result, List<Integer> list) {
      if (list.size() == nums.length) {
          //找到一个解，加入result（注意：这里要重新new一个list对象）
          result.add(new ArrayList<>(list));
          return;
      }
      //从0开始搜索
      for (int i = 0; i < nums.length; i++) {
          //判断该点是否已被访问过
          if (visited[i]) {
              continue;
          }
          //把该点加入list
          list.add(nums[i]);
          //把该点标记为已被访问
          visited[i] = true;
          //开始下一次搜索
          backtracking(nums, visited, result, list);
          //从该点退出，把该点标记为未被访问，并从list删除该点
          visited[i] = false;
          list.remove(list.size() - 1);
      }
  }
  
  ```

  ```java
  /**
  * https://leetcode.com/problems/permutations/
  * 方式二：把已访问过的点交换到前面去，防止被重复访问
  **/
  public List<List<Integer>> permute(int[] nums) {
      List<List<Integer>> result = new LinkedList<>();
      List<Integer> list = new LinkedList<>();
      //对解空间进行搜索
      backtracking(nums,result,0,list);
      return result;
  }
  /**
  * 搜索解空间
  **/
  public void backtracking(int[] nums, List<List<Integer>> result,
                           int start, List<Integer> list) {
      if(list.size()==nums.length) {
           //找到一个解，加入result（注意：这里要重新new一个list对象）
          result.add(new ArrayList<>(list));
          return;
      }
      //从start开始一次递归
      for(int i=start;i<nums.length;i++) {
          //把当前值nums[i]加入list
          list.add(nums[i]);
          //把该点与start进行交换（防止被重复访问）
          swap(nums,i,start);
          //进入下一个点
          backtracking(nums,result,start+1,list);
          //从该点退出，把该点与start换回去，并从list中删除
          swap(nums,i,start);
          list.remove(list.size()-1);
      }
  }
  /**
  * 交换i,j
  **/
  private void swap(int[] nums, int i, int j) {
      int tem = nums[i];
      nums[i] = nums[j];
      nums[j] = tem;
  }
  ```

  ```java
  /**
  * https://leetcode.com/problems/combination-sum-ii/
  * 该题也属于排列问题，但要对结果进行去重（也可以用一个HashSet进行去重，但不太优雅）
  **/
  public List<List<Integer>> combinationSum2(int[] candidates, int target) {
      List<List<Integer>> result = new LinkedList<>();
      List<Integer> list = new LinkedList<>();
      boolean[] visited = new boolean[candidates.length];
      //对数组进行排序，以便下一步对结果进行去重
      Arrays.sort(candidates);
      //进行搜索
      backtracking(candidates, result, visited, 0, list, 0, target);
      return result;
  }
  public void backtracking(int[] candidates, List<List<Integer>> result,
                           boolean[] visited,int start,
                           List<Integer> list, int curVal,int target) {
      if (curVal == target) {
          //找到一个答案
          result.add(new LinkedList<>(list));
          return;
      }
      //剪枝操作
      if (curVal > target) {
          //因为数组元素都是正数，所以该分支下不可能存在解
          return;
      }
      for (int i = start; i < candidates.length; i++) {
          //进行去重，该点前面有一个与其相等的值但没被访问，则continue
          if(i>0&&candidates[i]==candidates[i-1]&&!visited[i-1]) {
              continue;
          }
          //访问该点操作
          list.add(candidates[i]);
          curVal += candidates[i];
          visited[i] = true;
          backtracking(candidates, result,visited, i + 1, list, curVal, target);
          //退出该点操作
          list.remove(list.size() - 1);
          curVal -= candidates[i];
          visited[i] = false;	
      }
  }
  ```

  

+ 树

  类型：二叉树、2-3查找树、红黑树、B-数、B+数、平衡二叉树

  遍历：前序、中序、后序、层次

  算法：路径和、最小/大高度、公共祖先节点

+ 图

  类型：有向图、无向图

  数据结构：邻接矩阵、邻接表

  遍历：广度优先，深度优先

  算法：拓扑排序、最短路径、最小生成树、并查集

+ 深度优先搜索DFS

  类似回溯法，区别在于回溯法有一个”往回走“（回溯）的过程（如迷宫问题），而DFS更多的在于搜索(一直往下走，直到无路可走；然后直接回到起点，没有回溯过程)所有解集。

  **基本模板**（对比回溯法，少了回溯的操作，即在探索完后不用退回到上一个点）

  ```java
  public void function() {
      //选择点，进行搜索
      for() {
          dfs();
      }
  }
  private void dfs() {
      //递归终止条件
      if() {
          return;
      }
      //移到下一个点进行搜索
      for() {
          //进入前操作（一般是标记当前点为已访问）
       	//进入下一个点
          dfs();
      }
  }
  ```

  并查集问题

  ```java
  /**
  * https://leetcode.com/problems/max-area-of-island/
  **/
  public int maxAreaOfIsland(int[][] grid) {
      if(grid==null||grid.length<1) {
          return 0;
      }
      boolean[][] visited = new boolean[grid.length][grid[0].length];
      int maxSize = 0;
      for(int row=0;row<grid.length;row++) {
          for(int col=0;col<grid[0].length;col++) {
              //当前点没被访问，且值为1
              if(!visited[row][col]&&grid[row][col]==1) {
                  size = 0;
                  dfs(grid, visited,row,col);
                  maxSize = Math.max(maxSize,size);
              }
          }
      }
      return maxSize;
  }
  private int[][] directs = {{1,0}, {-1,0}, {0,1}, {0, -1}};
  private int size = 0;
  private void dfs(int[][] grid, boolean[][] visited,
                   int curRow, int curCol) {
      for(int[] direct : directs) {
          //在当前点的一些操作
          if(!visited[curRow][curCol]) {
              visited[curRow][curCol] = true;
              size++;
          }
          //进入下一个点
          int nextRow = curRow + direct[0];
          int nextCol = curCol + direct[1];
          //判断下一个点是否可以进入
          if(nextRow>=0&&nextRow<grid.length
            &&nextCol>=0&&nextCol<grid[0].length
            &&!visited[nextRow][nextCol]
            &&grid[nextRow][nextCol]==1) {
              //进入下一个点
              dfs(grid,visited,nextRow,nextCol);
          }
      }
  }
  ```

+ 广度优先搜索BFS

  用一个队列Queue实现**广度优先搜索模板**

  ```java
  public void function() {
      //定义一个队列
      LinkedList<?> queue = new LinkedList<>();
      //把起始点加入queue
      queue.offer(startPoint);
      while(!queue.isEmpty()) {
          //当前层次大小
          int size = queue.size();
          //访问该层的所有点
          while( size-- >0 ) {
              //对当前点进行一些操作
              Object cur = queue.poll();
              //判断该点是否满足条件
              if() {  
              }
              //从该点可以直接到达的点加入queue
              for() {
                  queue.offer(next);
              }
          }
      }
  }
  
  ```

  树的层次遍历

  ```java
  /**
  * https://leetcode.com/problems/binary-tree-level-order-traversal/
  **/
  public List<List<Integer>> levelOrder(TreeNode root) {
      List<List<Integer>> result = new LinkedList<>();
      if(root==null) {
          return result;
      }
      //定义一个队列
      LinkedList<TreeNode> queue = new LinkedList<>();
      //把起始点加入queue
      queue.offer(root);
      while(!queue.isEmpty()) {
          //当前层次大小
          int size = queue.size();
          List<Integer> list = new LinkedList<>();
          while( size-- > 0) {
              //访问当前点
              TreeNode node = queue.poll();
              list.add(node);
              //从该点可以直接到达的下一个点
              if( node.left != null) {
                  queue.offer(node.left);
              }
              if( node.right != null) {
                  queue.offer(node.right);
              }
          }
          result.add(list);
      }
      return result;
  }
  ```

  最短路径

  ```java
  /**
  * https://leetcode.com/problems/rotting-oranges/
  **/
  public int orangesRotting(int[][] grid) {
      if(grid == null||grid.length<1||grid[0].length<1) {
          return 0;
      }
      LinkedList<int[]> queue = new LinkedList<>();
      //把起始点加入queue
      for(int row=0;row<grid.length;row++) {
          for(int col=0;col<grid[0].length;col++) {
              if(grid[row][col]==2) {
                  queue.offer(new int[]{row,col});
              }
          }
      }
      int time = 0;
      //移动方向
      int[][] directs = {{1,0},{-1,0},{0,1},{0,-1}};
      while(!queue.isEmpty()) {
          int size = queue.size();
          boolean hasRotten = false;
          while(size-- > 0) {
              int[] curPos = queue.poll();
              int curRow = curPos[0];
              int curCol = curPos[1];
              //进入直接相邻的点
              for(int[] direct : directs) {
                  int nextRow = curRow + direct[0];
                  int nextCol = curCol + direct[1];
                  if(nextRow>=0&&nextRow<grid.length
                    &&nextCol>=0&&nextCol<grid[0].length
                     &&grid[nextRow][nextCol]==1) {
                      //把下一个点加入queue
                      grid[nextRow][nextCol] = 2;
                      queue.offer(new int[]{nextRow,nextCol});
                      hasRotten = true;
                  }
              }
          }
          if(hasRotten) {
              time++;
          }
      }
      //判断是否还有等于1的点
      for(int row=0;row<grid.length;row++) {
          for(int col=0;col<grid[0].length;col++) {
              if(grid[row][col]==1) {
                return -1;
              }
          }
      }
      return time;
  }
  ```

+ 栈数据结构Stack

  可以把递归实现转为非递归实现

  括号匹配

  ```java
  /**
  * https://leetcode.com/problems/valid-parentheses/
  **/
  public boolean isValid(String s) {
      char[] chs = s.toCharArray();
      LinkedList<Character> stack = new LinkedList<>();
      for (char ch : chs) {
          switch (ch) {
              case '{':
                  stack.add(ch);
                  break;
  			case '(':
  				stack.add(ch);
  				break;
  			case '[':
  				stack.add(ch);
  				break;
  			case '}':
  				if (stack.isEmpty() || stack.peek() != '{') {
  					return false;
  				}
  				stack.pop();
  				break;
  			case ')':
  				if (stack.isEmpty() || stack.peek() != '(') {
  					return false;
  				}
                  stack.pop();
                  break;
              case ']':
  				if (stack.isEmpty() || stack.peek() != '[') {
                      return false;
  				}
                  stack.pop();
  				break;
              default:
                  return false;
          }
      }
      return stack.isEmpty();
  }
  ```

  （使用两个栈Stack）

  中缀表达式的计算

  ```java
  /**
  * https://leetcode.com/problems/basic-calculator-ii/
  * 一个栈保存表达式中的数字，另一个栈保存表达式中的运算符。（重点是弄清楚进行运算符的优先级关系）
  **/
  class Solution {
      public int calculate(String s) {
          //存放数字的栈
          LinkedList<Integer> numStack = new LinkedList<>();
          //存放运算符的栈
  		LinkedList<Character> operaStack = new LinkedList<>();
          //运算符的优先级
  		int[] priority = new int[10];
  		priority[3] = 0;
  		priority[5] = 0;
  		priority[2] = 1;
  		priority[7] = 1;
  		boolean canCompute = false;
          for (int i = 0; i < s.length(); i++) {
              char now = s.charAt(i);
              //去除空格
  			if(now==' ') {
  				continue;
  			}
  			if (isDigit(now)) {
                  int num = 0;
                  //提取表达式中的数字
                  while (isDigit(now)) {
                      num = 10 * num + (now - '0');
                      i++;
                      if (i >= s.length()) {
  						break;
  					}
  					now = s.charAt(i);
  				}
                  //回退一个位置
  				i--;
  				numStack.push(num);
  				if (canCompute) {
  					int num1 = numStack.poll();
  					int num2 = numStack.poll();
  					char opera = operaStack.poll();
  					int value = compute(num2, num1, opera);
  					numStack.push(value);
  					canCompute = false;
                  }
              } else {
                  if (!operaStack.isEmpty()) {
  					char pre = operaStack.peek();
                      //当前优先级比之前运算符优先级小，则可以计算之前的表达式
  					if (priority[pre-40] >=priority[now-40]) {
  						int num1 = numStack.poll();
  						int num2 = numStack.poll();
  						int num = compute(num2, num1, pre);
  						numStack.push(num);
  						operaStack.poll();
  						operaStack.push(now);
  					} else {
  						operaStack.push(now);
  						canCompute = true;
  					}
  				} else {
  					operaStack.push(now);
  				}
  			}
  		}
  		int num1 = numStack.poll();
  		if(numStack.isEmpty()) {
  			return num1;
  		}
          int num2 = numStack.poll();
  		int num = compute(num2, num1, operaStack.poll());
  		return num;
  	
      }
      private int compute(int num1, int num2, char opera) {
          switch (opera) {
              case '+':
                  return num1 + num2;
              case '-':
                  return num1 - num2;
              case '*':
                  return num1 * num2;
              case '/':
                  return num1 / num2;
              default:
                  return -1;
          }
      }
      private boolean isDigit(char ch) {
          if (48 <= ch && ch <= 57) {
              return true;
          }
          return false;
      }
  }
  ```

  解压缩字符串（decode string）

  ```java
  /**
  * https://leetcode.com/problems/decode-string/
  * 一个数字栈，一个字符串栈
  **/
  public String decodeString(String s) {
      if (null == s || s.length() < 2) {
          return s;
      }
      LinkedList<Integer> numStack = new LinkedList<>();
      LinkedList<String> strStack = new LinkedList<>();
      int digit = 0;
      for (char ch : s.toCharArray()) {
          if (Character.isDigit(ch)) {
              digit = 10 * digit + (ch - '0');
          } else {
              if (digit != 0) {
                  numStack.push(digit);
              }
              digit = 0;
          }
          if (ch == '[') {
              strStack.push(ch + "");
          } else if (ch == ']') {
              String str = "";
              while (!strStack.isEmpty() && !strStack.peek().equals("[")) {
                  str = strStack.pop() + str;
              }
              int count = numStack.pop();
              String res = "";
              while (count-- > 0) {
                  res += str;
              }
              strStack.pop();
              strStack.push(res);
          } else if(!Character.isDigit(ch)){
              if (!strStack.isEmpty() && !strStack.peek().equals("[")) {
                  String str = strStack.pop();
                  str += ch;
                  strStack.push(str);
              } else {
                  strStack.push(ch + "");
              }
          }
      }
      StringBuilder str = new StringBuilder();
      while (!strStack.isEmpty()) {
          str.insert(0, strStack.pop());
      }
      return str.toString();
  }
  ```

+ 双指针

  两数之和

  ```java
  /**
  * https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
  **/
  public int[] twoSum(int[] numbers, int target) {
      int left = 0;
      int right = numbers.length-1;
      while(left<right) {
          int sum = numbers[left] + numbers[right];
          if(sum>target) {
              right--;
          } else if(sum<target) {
              left++;
          } else {
              return new int[]{left+1,right+1};
          }
      }
      return new int[]{-1,-1};
  }
  ```

  （快慢指针）

  链表的环问题

  ```java
  /**
  * https://leetcode.com/problems/linked-list-cycle/
  **/
  public boolean hasCycle(ListNode head) {
      if(null == head) {
          return false;
      }
      ListNode fast = head;
      ListNode slow = head;
      while(fast!=null&&slow!=null) {
          fast = fast.next;
          slow = slow.next;
          if(fast != null) {
              fast = fast.next;
          }
          if(fast == slow&&fast!=null) {
              return true;
          }
      }
      return false;
  }
  ```

  

+ 二分查找

  一般对有规律的数组进行查找，关键是发现一个能排除解空间的策略

  ```java
  /**
  * 目标值target在nums中的位置
  * https://leetcode.com/problems/binary-search/
  **/
  public int search(int[] nums, int target) {
      //对于无序可以先进行排序操作
      //Arrays.sort(nums);
      int left = 0;
      int right = nums.length-1;
      while(left <= right) {
          int midIdx = (left+right)>>>1;
          //排除解空间策略
          if(nums[midIdx] < target) {
              left = midIdx + 1;
          } else if(nums[midIdx] > target) {
              right = midIdx - 1;
          } else {
              return midIdx;
          }
      }
      return -1;
  }
  
  ```

  ```java
  /**
  * https://leetcode.com/problems/koko-eating-bananas/
  **/
  public int minEatingSpeed(int[] piles, int H) {
      //确定解空间范围最少吃1根，最多吃maxs(piles)根
      int left = 1;
      int right = maxs(piles);
      while(left < right) {
          //一个中间值
          int mid = (left+right)>>>1;
         //每小时吃mid根，吃完所需时间（抓住核心：全部吃完）
          int time = eat(piles,mid);
          //根据该值判断解在[left,mid]还是在(mid,right]范围
          if(time <= H) {
              //H小时内能吃完，下一次吃少点
              right = mid;
          } else if(time > H) {
              //H小时内吃不完，下一次吃多点
              left = mid + 1;
          }
      }
      return left;
  }
  /**
  * 每小时吃count根，全部吃完需要的时间
  **/
  private int eat(int[] piles, int count) {
      int time = 0;
      for(int num : piles) {
          time += num/count;
          //剩下的下一小时吃
          if(num%count != 0) {
              time++;
          }
      }
      return time;
  }
  /**
  * 返回数组nums中的最大值
  **/
  private int maxs(int[] nums) {
      int max = nums[0];
      for(int num : nums) {
          max = Math.max(max,num);
      }
      return max;
  }
  ```

  

+ 堆数据结构（Java中的PriorityQueue可以作为堆使用）

  堆在存储结构上是一个数组，逻辑结构是一颗完全二叉树

  ```java
  /**
  * 实现一个int类型大顶堆
  * （进一步：使用泛型以支持更多的数据类型、用比较器兼容大/小顶堆、一些边界的处理）
  **/
  public class MaxHeap {
      private final ArrayList<Integer> data;
      public MaxHeap() {
          this(11);
      }
      public MaxHeap(int capacity) {
          data = new ArrayList<>(capacity);
      }
      public void add(Integer t) {
          //把元素添加到数组最后
          data.add(t);
          //进行上浮操作
          swim(data.size()-1);
      }
      /**
      * 获取堆顶值
      **/
      public int find() {
          return data.get(0);
      }
      /**
      * 删除堆顶值，并返回
      **/
      public int delete() {
          int val = data.get(0);
          //删除堆顶元素，把最后一个元素放在堆顶
          data.set(0,data.get(data.size()-1));
          data.remove(data.size()-1);
          //进行下沉操作
          sink(0);
          return val;
      }
      /**
      * 当前节点如果比其子节点小，则进行交换（和最大的那个进行交换），
      * 并进行下一次迭代；否则返回
      **/
      private void sink(int k) {
          //当前节点有左子节点
          while(leftChild(k)<data.size()) {
              //选择一个左子节点作为待交换节点
              int choosedIdx = leftChild(k);
              //右子节点存在，且比其左子节点大
              if(rightChild(k) < data.size()
                 &&data.get(rightChild(k))>data.get(choosedIdx)) {
                  choosedIdx = rightChild(k);
              }
              //当前节点比其子节点大，则break
              if(data.get(k)>=data.get(choosedIdx)) {
                  break;
              }
              swap(k,choosedIdx);
              k = choosedIdx;
          }
      }
      /** 
      * 当前节点值如果比其父节点值大，则进行交换，并进行下一次迭代；否则返回
      **/
      private void swim(int k) {
          //当前节点不是根节点，且值比其父节点大
          while(k > 0 && data.get(k)>data.get(parent(k))) {
              swap(k,parent(k));
              k = parent(k);
          }
      }
      /**
      * 返回节点idx的父节点下标
      **/
      private int parent(int idx) {
          return (idx-1)/2;
      }
      /**
      * 返回idx对应的左孩子下标
      **/
      private int leftChild(int idx) {
          return 2 * idx + 1;
      }
      /**
      * 返回idx对应的右孩子下标
      **/
      private int rightChild(int idx) {
          return 2 * idx + 2;
      }
      /**
      * 交换i，j下标对应的值
      **/
      private void swap(int i, int j) {
          Integer tem = data.get(i);
          data.set(i,data.get(j));
          data.set(j,tem);
      }
  }
  ```

  解决TopK问题

  ```java
  /**
  *https://leetcode.com/problems/kth-largest-element-in-an-array/
  **/
  public int findKthLargest(int[] nums, int k) {
      MaxHeap heap = new MaxHeap();
      for (int num : nums) {
          heap.add(num);
      }
      int val = 0;
      while (k-- > 0) {
          val = heap.delete();
      }
      return val;
  }
  ```

+ 链表问题

  反转链表、链表求和、环、链表排序

  （要特别注意指针）

- 动态规划

  明确：状态（父子问题的关联）、状态转移方程、初始条件和边界

  一般是把问题规模缩小，通过解决子问题，最后进行迭代从而解决父问题。（**难点在于找到状态转移方程**）

  ```java
  /**
  * https://leetcode.com/problems/climbing-stairs/
  * 状态转移方程：f(n) = f(n-1) + f(n-2)
  * 初始解：f(0) = 0; f(1) = 1; f(2) = 2
  * 方式一：根据状态转移方程和初始解，利用一个数组直接计算
  **/
  public int climbStairs(int n) {
      if( n<0 ) {
          return -1;
      }
      if( n <= 2 ) {
          return n;
      }
      int[] fn = new int[n+1];
      fn[1] = 1;
      fn[2] = 2;
      for(int i=3;i<=n;i++) {
          fn[i] = fn[i-1] + fn[i-2];
      }
      return fn[n];
  }
  ```

  ```java
  /**
  * https://leetcode.com/problems/climbing-stairs/
  * 状态转移方程：f(n) = f(n-1) + f(n-2)
  * 初始解：f(0) = 0; f(1) = 1; f(2) = 2
  * 方式二：状态转移方程中，只用到了前两个状态值，所以可以进行空间优化
  **/
  public int climbStairs(int n) {
      if( n<0 ) {
          return -1;
      }
      if( n <= 2 ) {
          return n;
      }
      int f1 = 1;
      int f2 = 2;
      for(int i=3;i<=n;i++) {
          f2 = f1 + f2;
          f1 = f2 - f1;
      }
      return f2;
  }
  ```

  ```java
  /**
  * https://leetcode.com/problems/unique-paths-ii/
  * 状态转移方程：dp[m][n] = dp[m-1][n]+dp[m][n-1]
  * (这题只用到了dp左边和上边的值，所以同样可以对空间进行优化，把dp从二维数组转为一维数组)
  **/
  public int uniquePathsWithObstacles(int[][] obstacleGrid) {
      if (obstacleGrid == null || obstacleGrid.length == 0
          || obstacleGrid[0][0] == 1) {
          return 0;
      }
      int maxRow = obstacleGrid.length;
      int maxCol = obstacleGrid[0].length;
      int[][] dp = new int[maxRow][maxCol];
      //初始化初始状态
      dp[0][0] = 1;
      for (int col = 1; col < maxCol; col++) {
          if (obstacleGrid[0][col] == 0) {
              dp[0][col] = dp[0][col - 1];
          } else {
              dp[0][col] = 0;
          }
      }
      for (int row = 1; row < maxRow; row++) {
          if (obstacleGrid[row][0] == 0) {
              dp[row][0] = dp[row - 1][0];
          } else {
              dp[row][0] = 0;
          }
      }
      //利用状态转移方程进行迭代求解父问题
      for (int row = 1; row < maxRow; row++) {
          for (int col = 1; col < maxCol; col++) {
              if (obstacleGrid[row][col] == 1) {
                  dp[row][col] = 0;
              } else {
                  dp[row][col] = dp[row - 1][col] + dp[row][col - 1];
              }
          }
      }
      return dp[maxRow - 1][maxCol - 1];	
  }
  ```

  ```java
  /**
  * https://leetcode.com/problems/unique-paths-ii/
  * 状态转移方程：dp[m][n] = dp[m-1][n]+dp[m][n-1]
  * 用一维数组dp进行实现
  **/
  public int uniquePathsWithObstacles(int[][] obstacleGrid) {
      if (obstacleGrid == null || obstacleGrid.length == 0
          || obstacleGrid[0][0] == 1) {
          return 0;
      }
      int maxRow = obstacleGrid.length;
      int maxCol = obstacleGrid[0].length;
      int[] dp = new int[maxCol];
      dp[0] = 1;
      for (int row = 0; row < maxRow; row++) {
         for(int col=0; col < maxCol; col++) {
             if(obstacleGrid[row][col] == 1) {
                 dp[col] = 0;
             }else {
                 if( col==0 ) {
                     dp[col] = dp[col];
                 } else {
                     //dp[col] = dp[col]+dp[col-1]
                     dp[col] += dp[col-1];
                 }
             }
         }
      }
      return dp[maxCol - 1];	
  }
  ```

- 贪心算法

  找到贪心策略，按照贪心策略依次进行计算，可以大大减少讨论的情况

  ```java
  /**
  * https://leetcode.com/problems/assign-cookies/
  * 贪心策略是尽量用小的cookie满足要求小的child
  **/
  public int findContentChildren(int[] g, int[] s) {
      Arrays.sort(g);
      Arrays.sort(s);
      int childIdx = 0;
      int cookieIdx = 0;
      int count = 0;
      for(;childIdx<g.length;childIdx++) {
          for(;cookieIdx<s.length;cookieIdx++) {
              if(s[cookieIdx]>=g[childIdx]) {
                  cookieIdx++;
                  count++;
                  break;
              }
          }
      }
      return count;
  }
  ```

  ```java
  /**
  * https://leetcode.com/problems/non-overlapping-intervals/
  * 转为找数组中最多不重叠的数量
  **/
  public int eraseOverlapIntervals(int[][] intervals) {
      //边界输入判断
      if (null == intervals||intervals.length==0) {
          return 0;
      }
      //按照终点正序序，起始点逆序排列（优先选择终点小起始点大的，尽量缩小被选择序列的占空间）
      Arrays.sort(intervals, new Comparator<int[]>() {
          @Override
          public int compare(int[] o1, int[] o2) {
              if(o1[1]>o2[1]) {
                  return 1;
              }
              if(o1[1]<o2[1]) {
                  return -1;
              }
              if(o1[0]>o2[0]) {
                  return -1;
              }
              if(o1[0]<o2[0]) {
                  return 1;
              }
              return 0;
          }
      });
      //使用贪心策略，依次查找
      int end = intervals[0][1];
      int count = 1;
      for(int i=1;i<intervals.length;i++) {
          if(intervals[i][0]>=end) {
              count++;
              end = intervals[i][1];
          }
      }
      return intervals.length-count;
  }
  ```

- 滑动窗口

  两次计算有重叠部分时，可以考虑用滑动窗口方法以减少时间复杂度。

  一般计算公式：当前值+右边新移入的值-左边窗口移出的值

  ```java
  /**
  * https://leetcode.com/problems/grumpy-bookstore-owner/
  **/
  public int maxSatisfied(int[] customers, int[] grumpy, int X) {
      int sum = 0;
      //可以满足所有人
      if(X >= customers.length) {
          for(int i=0;i<customers.length;i++) {
              sum += customers[i];
          }
          return sum;
      }
      //原先能满意的用户数量
      for(int i=0;i<customers.length;i++) {
          sum+=(1-grumpy[i])*customers[i];
      }
      //遍历区间
      int left = 0;
      int right = X-1;
      //现增加的用户
      int curIncNums = 0;
      //计算第一个窗口值
      for(int i=left;i<=right;i++) {
          curIncNums += grumpy[i]*customers[i];
      }
      int maxIncNums = curIncNums;
      left++;
      right++;
      //窗口向右滑动
      for(;right<customers.length;right++,left++) {
          curIncNums = curIncNums-
              grumpy[left-1]*customers[left-1]+grumpy[right]*customers[right];
          maxIncNums = Math.max(maxIncNums, curIncNums);
      }
      return sum+maxIncNums;
  }
  ```

+ 其他

  一致性Hash算法（用TreeMap可以实现）

  哈夫曼编码（用PriorityQueue可以实现）

  洗牌算法

  按权重随机负载均衡

  背包问题

  二分图问题

  分而治之
