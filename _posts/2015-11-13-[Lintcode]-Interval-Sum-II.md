---
layout: post
title:  "[Lintcode] Interval Sum II"
tags:
- algorithm
- interview
---

> Given an integer array in the construct method, implement two methods query(start, end) and modify(index, value):
* For query(start, end), return the sum from index start to index end in the given array.
* For modify(index, value), modify the number in the given index to value

{% highlight java %}
public class Solution {

    private TreeNode root;

    public class TreeNode {
        int start, end;
        long sum;
        TreeNode left, right;
        public TreeNode(int start, int end, long sum) {
            this.start = start;
            this.end = end;
            this.sum = sum;
            this.left = this.right = null;
        }
    }

    public Solution(int[] A) {
        int n = A.length;
        if (n == 0)
            root = null;
        else
            root = build(A, 0, n-1);
    }

    private TreeNode build(int[] A, int start, int end) {
        if (start == end) {
            return new TreeNode(start, end, A[start]);
        } else {
            int mid = (end - start) / 2 + start;
            TreeNode left = build(A, start, mid);
            TreeNode right = build(A, mid+1, end);
            TreeNode root = new TreeNode(start, end, left.sum + right.sum);
            root.left = left;
            root.right = right;
            return root;
        }
    }

    public long query(int start, int end) {
        return query(root, start, end);
    }

    public long query(TreeNode root, int start, int end) {
        if (root == null) {
            return 0;
        } else {
            int left = root.start, right = root.end;
            if (left >= start && right <= end) {
                return root.sum;
            } else {
                int mid = (right - left) / 2 + left;
                if (end <= mid)
                    return query(root.left, start, end);
                else if (start > mid)
                    return query(root.right, start, end);
                else
                    return query(root.left, start, mid) + query(root.right, mid+1, end);
            }
        }
    }

    public void modify(int index, int value) {
        modify(root, index, value);
    }

    private void modify(TreeNode root, int index, int value) {
        if (root == null) {
            return;
        } else {
            int left = root.start, right = root.end;
            if (left == index && right == index) {
                root.sum = (long)value;
            } else {
                int mid = (right - left) / 2 + left;
                if (index <= mid)
                    modify(root.left, index, value);
                else
                    modify(root.right, index, value);
                long sum = (root.left == null) ? 0 : root.left.sum;
                sum += ((root.right == null) ? 0 : root.right.sum);
                root.sum = sum;
            }
        }
    }
}
{% endhighlight %}
