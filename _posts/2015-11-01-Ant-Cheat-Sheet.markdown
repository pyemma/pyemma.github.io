---
layout: single 
title:  "Ant Cheat Sheet"
tags:
- project
- tools
---
This is a simple document introducing how to write the build.xml of Ant for your projects.

Each Ant build.xml has a `<project>` element as the root element, this element can have a `name` attribute, which specify the name of this project; a `basedir` element, which determines the root path during the sequential path calculation, a `default` element, which is the default `target` to run.

Each `<project>` can be constructed with several `<target>`,  `<target>` can be think of as each state of our application. For example, in my project, I need to first create several dictionaries, and then compile some java class, and encapsulate the compiled class into jar file, and then run some test. All these can be regarded as individual `<target>`. `<target>`  also has `name` attribute and it can also be specified with `depends` attribute which is other `<target>` that this depends on.

Each `<target>` is composed of several `task`. The `task` is in the format like `<task_name attr1=val1 attr2=val2 />`. Several `task` that I have used is

* **mkdir**: create dictionaries
* **delete**: delete a file or a dictionary
* **javac**: used to comple java class file
* **java**: run java class
* **jar**: create jar file
* **exec**: run a executable binary file

Sometimes, we need the same value multiple times in different places, like the name of dictionaries. We can use `<property>` to hold this value and use the `<property>` in other place. For example, `<property name="build.dir" value="build" />`, then we can use it like `<java srcdir="${src.dir}" dstdir="${build.dir}" />`.

When compile or run java class, we sometimes need external libraries, we can use `<path>` or `<classpath>` to hold them. For example
{% highlight xml %}
<path id="classpath">
	<pathelement path="${classpath}" />
    <fileset dir="${lib.dir}" includes="**/*.jar"/>
</path>
{% endhighlight %}
Here we assigned an id to this `<path>` element, we can later reference this id. And there can be multiple `<pathelement>`, the `path` attribute is  usually used with predefined path and `location` specify a path relative to the base path.
{% highlight xml %}
<target name="run" depends="jar">
    <java fork="true" classname="${main-class}">
        <classpath>
            <path refid="classpath"/>
            <path location="${jar.dir}/${ant.project.name}.jar"/>
        </classpath>
    </java>
</target>
{% endhighlight %}
We can use `<echo>` in `<target>` to output some information.

> Useful link: [Ant manual](https://ant.apache.org/manual/)
