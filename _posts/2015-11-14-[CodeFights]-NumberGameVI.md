---
layout: post
title:  "[CodeFights] NumberGameVI"
tags:
- algorithm
- interview
---

The problem is available [here](https://codefights.com/challenge/2HDYBD5D7btqPZqqb). A relatively straight forward game problem. If the current person want to win, the only requirement is that there exist a move can make B lose. And if there is no move can make B lose, then A will lose. Translate this into recursion and then we can have a working code. We can use a record table to hold whether we can win or not from the current state, using this memory to reduce the required recursion.

{% highlight java%}
HashMap<String, Integer> record = new HashMap<String, Integer>();

int NumberGameVI(int a, int b, int c, int d) {
    String str = a + " " + b + " " + c + " " + d;
    if (record.containsKey(str) == true)
        return record.get(str);
    for (int num = 1; num < c; ++num) {
        if ((num != a ) && (num != b) && (num != c)) {
            int win = -1;
            if (num < a) {
                win = NumberGameVI(num, a, b, c);
            } else if (num < b) {
                win = NumberGameVI(a, num, b, c);
            } else {
                win = NumberGameVI(a, b, num, c);
            }
            if (win == -1) {
                record.put(str, num);
                return num;
            }
        }
    }
    record.put(str, -1);
    return -1;
}
{% endhighlight %}

> Similar Problem: Still two person A and B, they take values from a given list (all positive values) in turns. The first person who make the total sum above a given target can win.
