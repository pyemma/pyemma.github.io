---
layout: single
title:  "[Lintcode] Count of Smaller Number before itself"
tags:
- algorithm
- interview
---

> Give you an integer array (index from 0 to n-1, where n is the size of this array, value from 0 to 10000) . For each element `Ai` in the array, count the number of element before this element `Ai` is smaller than it and return count number array.

In this problem, we need to first build an big enough (0 to 10000) segment tree which hold the count of elements in each range. Then as we go along the array, we update the segment tree and query the segment tree. At first, I was trying to **adding** new tree node into this segment tree, but this method does not work, since it needs to **rearrange** the segment tree to make it stay in a consistent state. So the better way is to initialize a big enough tree first and then modify on it.
{% highlight java %}
public class Solution {

    public ArrayList<Integer> countOfSmallerNumberII(int[] A) {
        // write your code here
        TreeNode root = build(0, 10001);
        ArrayList<Integer> result = new ArrayList<Integer>();
        for (int i = 0; i < A.length; ++i) {
            result.add(query(root, 0, A[i]-1));
            modify(root, A[i]);
        }
        return result;
    }

    public class TreeNode {
        int start, end, count;
        TreeNode left, right;
        public TreeNode(int start, int end, int count) {
            this.start = start;
            this.end = end;
            this.count = count;
        }
    }

    public int query(TreeNode root, int start, int end) {
        if (root == null) {
            return 0;
        } else {
            int left = root.start, right = root.end;
            if (start <= left && end >= right)
                return root.count;
            else {
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

    public TreeNode build(int start, int end) {
        if (start == end) {
            return new TreeNode(start, end, 0);
        } else {
            int mid = (end - start) / 2 + start;
            TreeNode root = new TreeNode(start, end, 0);
            root.left = build(start, mid);
            root.right = build(mid+1, end);
            return root;
        }
    }

    public void modify(TreeNode root, int value) {
        if (root == null) {
            return;
        } else {
            int left = root.start, right = root.end;
            if (left == value && right == value) {
                root.count += 1;
            } else {
                int mid = (right - left) / 2 + left;
                if (value <= mid)
                    modify(root.left, value);
                else
                    modify(root.right, value);
                root.count += 1;
            }
        }
    }
}
{% endhighlight %}
