---
layout: post
title:  "[Lintcode] Triangle Count" 
tags:
- algorithm
- interview
---
> Given an array of integers, how many three numbers can be found in the array, so that we can build an triangle whose three edges length is the three numbers that we find?

The most naive solution is to enumerate all triples and then check if they can construct a triangle or not. The time complexity would be $$O(n^3)$$. A better solution is to first sort the elements in the array, then enumerate the two shorter edge and use binary search to locate the position of the longest edge. The time complexity would be $$O(nlogn) + O(n^2logn) = O(n^2logn)$$.
{% highlight ruby %}
public int triangleCount(int S[]) {
    // write your code here
    int n = S.length;
    if (n < 3)
        return 0;
    Arrays.sort(S);
    
    int count = 0;
    for (int i = 0; i < n-2; ++i) {
        for (int j = i+1; j < n-1; ++j) {
            int index = Arrays.binarySearch(S, j+1, n, S[i] + S[j]);
            index = (index < 0) ? -(index + 1) : index;
            count += (index - j - 1);
        }
    }
    return count;
}
{% endhighlight %}
Related problem: [Hackerrank Counting Triangles](https://www.hackerrank.com/contests/codestorm/challenges/ilia)