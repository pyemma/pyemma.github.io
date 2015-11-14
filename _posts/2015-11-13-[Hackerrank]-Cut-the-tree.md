---
layout: post
title:  "[Hackerrank] Cut the tree"
tags:
- algorithm
- interview
---
The problem is available [here](https://www.hackerrank.com/challenges/cut-the-tree). The idea is to use postorder traversal of the tree to enumerate each possible remove of edge, and return the sum of the sub-tree to its parent node. This problem represent the tree as a acyclic undirected graph, which is a quite wired representation.
{% highlight java %}
public class Solution {

    public static class GraphNode {
        int id, val;
        List<GraphNode> neighbors;
        public GraphNode(int id, int val) {
            this.id = id;
            this.val = val;
            neighbors = new ArrayList<GraphNode>();
        }
    }

    static int result = Integer.MAX_VALUE;

    public static int dfs(GraphNode root, int sum, HashSet<Integer> visited) {
        if (root == null) {
            return 0;
        } else {
            int curSum = 0;
            visited.add(root.id);
            for (GraphNode neighbor : root.neighbors) {
                if (visited.contains(neighbor.id) == false) {
                    int res = dfs(neighbor, sum, visited);
                    // System.out.println(sum + " " + res);
                    result = Math.min(Math.abs(sum - 2 * res), result);
                    curSum += res;
                }
            }
            return curSum + root.val;
        }
    }

    public static void main(String[] args) {

        Scanner scan = new Scanner(System.in);
        int n = scan.nextInt();
        List<GraphNode> nodes = new ArrayList<GraphNode>();
        int sum = 0;
        for (int i = 0; i < n; ++i) {
            int val = scan.nextInt();
            nodes.add(new GraphNode(i+1, val));
            sum += val;
        }
        int a = scan.nextInt(), b = scan.nextInt();
        GraphNode root = nodes.get(a-1);
        nodes.get(a-1).neighbors.add(nodes.get(b-1));
        nodes.get(b-1).neighbors.add(nodes.get(a-1));
        for (int i = 0; i < n-2; ++i) {
            a = scan.nextInt();
            b = scan.nextInt();
            nodes.get(a-1).neighbors.add(nodes.get(b-1));
            nodes.get(b-1).neighbors.add(nodes.get(a-1));
        }
        HashSet<Integer> set = new HashSet<Integer>();
        dfs(root, sum, set);
        System.out.println(result);
    }
}
{% endhighlight %}
