# python

## 目录

```python
#                                                   目录
# 目录
#   命名规范
#       --
#   input()与print()函数的使用
#       input()
#       print()
#           直接使用%与sep、end进行格式化输出
#           使用{}与format方法进行格式化输出
#           使用f"字符串{变量名}"进行格式化输出
#   基本容器与数据类型（含常见API）
#       String
#       list
#       tuple
#       set
#       dict
#   函数
#       函数的定义（位置参数、关键字参数、形式参数）
#       变量在函数内外的作用域与global关键字
#       匿名函数lambda
#   多文件编写
#       库、包、模块的区别与联系
#       import函数
#       __name__属性
#   其他杂项
#       --
#   常见API
#       type
#       math
#       random
#       列表逆序
```



## 命名规范

```python
"""
********************************************************************************************





                                        命名规范





********************************************************************************************
"""
'''
    命名规范
        类名              大驼峰命名法（同java）
        变量名、函数名      全部小写，单词与单词之间用_连接
'''
```



## input()与print()函数的使用

```python
"""
********************************************************************************************





                                        input()与print()函数的使用





********************************************************************************************
"""
```

### input()

```python
"""
**********************************************************************************************
                                  inPut函数的使用
**********************************************************************************************
"""
def test_input(self):
    #   1、input函数的默认返回值是字符串类型，所以如果想要赋值的是整数类型，就要强制转换，但是不是（int）而是int(需要强制转换的内容)
    a = input("请输入一个数字,但是录入的时候，还是字符串类型")
    b = int(input("请输入一个数组，录入的时候就是整数int类型"))
```

### print()

```python
"""
**********************************************************************************************
                                  print函数的使用
**********************************************************************************************
"""
#   参考笔记链接    https://www.perfcode.com/python-built-in-functions/python-print.html
#               https://blog.csdn.net/weixin_69553582/article/details/125403845/
'''
    总结
        有关格式问题
            1)print中的逗号后要有空格，例如print("abc", "cde", "xyz")
                                而不是print("abc","cde","xyz")
                                这样可以减少警告
            2)print函数的原型是def print(*values: object,
                                      sep: str | None = " ",
                                      end: str | None = "\n",
                                      file: SupportsWrite[str] | None = None,
                                      flush: Literal[False] = False) -> None)
                    其中*values: object是一个列表，所以理论上可以装无穷多的东西
                        sep与end这样的属性为了与values中的object变量区别开来，所以必须写sep="--"，而不是直接"--"!
            3)正确示例  print("aaa", "bbb", sep="--", end="\n")
                        sep与end属性之间也是用,加空格实现
                        sep="--", 尽量不要写成sep = "--", 这里面不要加空格
            4)默认的情况下，print函数的末尾是用"\n",sep=" "
'''
def test_print(self):
    # 基本使用方式
    print("# 基本使用方式print(*values object, sep, end, file flush)")
    print("\t控制打印不换行", end="")
    print("\n", end="")
    print("\t默认字符串之间用空格隔开""因为默认的sep参数为空格")    # 这里没有用逗号，一样可以打印，但是没有视为两个字符串，中间没有分隔符sep
    print("\t默认字符串之间用空格隔开", "因为默认的sep参数为空格")   # 这里用了逗号，打印的时候就会把两个东西视为两个字符串，中间自带了默认的分隔符sep
    print()

    # 利用占位符进行格式化输出
    print("# 利用占位符进行格式化输出")
    age = 18
    goal = 95.573
    print("\t我的名字是%s, 今年%d岁，成绩是%.2f" % ("小明", age, goal))
    print()

    # 利用占位符与键值对（字典）来进行格式化输出
    print("# 利用占位符与键值对（字典）来进行格式化输出")
    name = "小明"
    print("\t我的名字是%(name)s, 今年%(age)d岁" % {"name": "小王", "age": 19})   # 这里键值对的值与作为键的变量本身的值无关，而是与字典中的值有关
    print()

    # 利用{}__format进行格式化输出
    '''
    format本质上是字符串的一个方法，用来将参数列表中的字符串填充到原来的字符串中，然后返回一个新的字符串，与字典无关，就仅仅是一个类的方法
    '''
    print("# 利用{}__format进行格式化输出")
    print("\tformat按照顺序输出")
    print("\t\t我的名字是{}，今年{}岁".format("hello", "world"))
    print("\t不按顺序，在{}中指定输出位置在format参数列表的下标")
    print("\t\t我的名字是{1}，今年{0}岁".format(25, "小张"))   # format参数中默认是args类型，所以数字会自动转化
    print("\t按照键值对的方式指定数据（注意，只是键值对，不是字典，所以不是大括号）")
    print("\t\t网站名:{name}, 网址是:{url}".format(name="百度", url="http:www.baidu.com"))
    print("\t用字符串的format方法填充浮点数")
    print("\t\t我的分数是{:.2f}".format(92.572))  # 一定要有：.的结构，这个冒号相当于格式化输出中的%
    print()

    # 利用f格式，直接进行格式化输出(注意，f要写在""外面)
    print("# 利用f格式，直接进行格式化输出(注意，f要写在""外面)")  # 这里出现了""中还有""的情况，最后结果是输出的是""空字符串，不需要单双引号昏混用（当然有可能是python版本的原因，支持混用了）
    goal2 = 95
    print(f"\t我的年龄是{age}")
    print(f"\t我的学校是{'电子科技大学'}")
    print(f"\t我的成绩是{95}")
    print(f"\t我的成绩是{goal2}")
    # f_string格式化输出的时候，{}里面可以是变量，也可以是常量
    # f应该本质上也是字符串的一个特殊的使用方法，而不是print的方法，但是底层原理还需要弄明白
```





## 基本容器与数据类型（含常见API）

```python
"""
********************************************************************************************





                                    基本容器与数据类型
                    字符串(String)、列表(list)、元组(tuple)、集合(Set)、字典(dict)
                    
                    



********************************************************************************************
"""
"""
                                        各个数据类型的比较
            字符串(String)     str = "abc"                                     不可更改
            列表(list)        list_a = [1, "abc", (1, 20)]                    ------- (可更改)
            元组(tuple)       tuple_a = (1, 2, "abc")                         不可更改
            集合(Set)         Set_a = {1, "abc", (1, 20)}                     ------- (可更改)
            字典(dict)        dict_a = {"name":uesct, "place":"Chengdu"}      ------- (可更改)
"""
```

### String

```python
"""
********************************************************************************************
                                        字符串的使用
********************************************************************************************
"""
'''
字符串的使用
    字符串的定义
        1）直接使用单引号与双引号定义
        2）使用三引号定义长文本（在注释里不好演示，见注释下方的测试函数）
    字符串的使用
        1）len(字符串)获取字符串的长度
        2）r"字符串"    r表示将字符串中的所有转义字符全部看成普通字符，不在需要\\n，直接是r"\n"最后输出的就是\n
        3)f"字符串{a}字符串",表示把a这个变量或者常量的值变成字符串加入到原字符串中（这个加入并不是改动了源字符串，而是创建了新的字符串）
    字符串的切取
        1)切取语法的格式为[start:end:step]
        2)start表示切取的开始位置，end表示切取的结束位置，step表示切取的步长和方向
        3)切出来的范围是[start, end)左闭右开
        4)默认start为0，end为len（即最后一个字符的后一个位置），step=1
        5)切片的时候start和end填充的索引值有正向索引和逆向索引两种，正向索引从左到右进行编号，第一个元素是0，这一点同数组相同
            逆向索引，从右到左进行标号，最后一个元素的下标为-1,之后依次记为-2,-3,-4……
        注意：
            如果step规定为负数的时候，就说明切取的方向是从右到左，所以，这个时候start要大于end
            start与end能不能一个采用正向索引，一个采用逆向索引？（可以试一下，感觉是可以的）
    字符串的常见api
        len(string) 返回字符串的长度
        string.count("字符串")返回目标字符串在调用方法的字符串中出现的次数
        string.capitalize()将首字母大写
        string.find("目标字符串")查找指定字符串在源字符串中是否出现，如果出现，返回第一次出现的索引值，如果出现，就返回-1
        string.index("目标字符串")，查找目标第一次出现的下标，与string.find()类似，但是默认是目标字符串是存在的，如果不存在是会报错的
        string.split("分隔符")，通过分隔符把源字符串分割为几个部分，装入列表中，分隔符可以是字符串，Python中字符与字符串都可以用双引号，单引号，没什么大的区别
        string.replace("src", "replacement", 替换的个数)，替换字符串，注意要写替换多少个
        
        string.upper()  把字符串中的字符全部变为大写
        string.lower()  把字符串中的字符全部变为小写
        string.startswith("字符串A")    判断字符串是否以字符串A开头
        string.endswith("字符串B")      判断字符串是否以字符串B开头
        string.strip("目标")      删除字符串两边的“目标”，如果没有传入参数，默认删除字符串两边的所有空格 
        string.lstrip("目标")     删除左边的目标
        string.rstrip("目标")     删除右边的目标
'''
# 用于测试的函数
def test_string(self):
    # 字符串的创建
    print("# 字符串的创建")
    s = '''
        长文本
        长文本
        长文本
    '''
    print("\t", s)

    # 字符串的切取
    print("# 字符串的切取")
    array = "我们的征途是星辰大海"
    print("\t全部切取：array[:]", array[:])
    print("\t正向不采用步长进行切取：array[1:4]", array[1:4])
    print("\t正向采用步长进行切取(给与start)：array[1::1]", array[1::1])
    print("\t正向采用步长进行切取：(给与end):array[:5:1]", array[:5:1])
    print("\t逆向不采用步长进行切取：(给与start与end) array[-2:-5:-1]", array[-2:-5:-1])
    print("\t逆向采用步长进行切取：(给与start与end) array[-2:-10:-2]", array[-2:-10:-2])
    print("\t正逆下标混用：array[1:-1]", array[1:-1])
```

### list

```python
"""
********************************************************************************************
                                        列表的使用
********************************************************************************************
"""
'''
列表的使用
    列表的本质：就是java中的集合容器，通过对列表元素的print(type())可以知道，他的结构是<class list>
    其他
        列表因为是容器，所以可以装下任何的引用数据类型
        列表可以支持拼接，c = [1, 2, 3, 4]
                      b = [5, 6, 7]
                      print(c+b)就是[1, 2, 3, 4, 5, 6, 7]
        列表的元素可以通过下标查找
        列表支持切片，语法同字符串的切片
        列表可以嵌套  eg.b = [1, 2, [3, 4, 5]]
                        3的下标就是b[2][0],这个与numpy中二维数组的访问是arr[2,0]不同，需要注意
    一些方法
        列表的乘法
            print(b*4)  表示列表中的数据复制成为四份
        判断某个元素是否在列表中
            print("a" in listB) 判断"a"是否在listB中，返回true与false
            
    常见的API
        构造方法（不同类型或者容器的转换）
            字符串转列表list(字符串)  # 返回构造方法创建的列表
        成员方法
            增
                listA.append(Object obj)    # 在列表尾部添加新的元素
                listA.extend(List list)     # 将list列表中的内容全部从尾部添加到listA中，这个与列表加法的区别是，列表的加法不改变原有的列表
                listA.insert(int index, Object obj)  #将obj元素插入到index下标处，原来的index位置的元素往后退 
            删
                listA.remove(Object obj)    # 删除指定元素obj，但是只删除找到的第一个
                listA.pop()                 # 空参pop函数，删除列表的最后一个元素
                # pop方法是有返回值的，返回的就是被删除的这个元素，例如在list中的pop()返回的就是list类型
                                                             在dict中的pop()返回的就是Object类型,键名对应的value值
                listA.pop(int index)        # 通过指定的下标删除元素
            查
                listA.count(Object obj)     # 判断obj是否在listA中存在
                listA.index(Object obj)     # 返回obj在listA中的第一次出现的下标值，如果没有出现就报错
                a in b
            其他
                listA.copy()                # 返回lsit类型，(列表),用于复制列表
                listA.reverse()             # 用于对列表反序
                listA.sort()                # 默认只能升序排列，只能作用于只含有数字的列表
                listA.sort(reverse=true)    # 降序排列，reverse是sort中的boolean类型的参数中，确认排序之后使用依次reverse
                list.sort(key=lambda x: 匿名函数内容，书写关于比较器的规则)
                                            # 这里key是sort函数的参数，就像qsort中的cmp一样
'''
```

### tuple

```python
"""
********************************************************************************************
                                        元组tuple
********************************************************************************************
"""
'''
元组
    单元素元组与int的区分
        单元素元素   (1,)    # 必须要有这个逗号，否则就成了int类型了
        元组与列表差不多，需要注意的是，元组不能修改，但是可以切片，因为切片本质上不改变元数据，只是返回了切片之后的新数据
        元组也可以通过有参构造把列表或者字符串转化为元组
        元素内部如果有列表作为元素，列表的值是可以修改的，这个与元素不可修改不矛盾，因为对于元组而言，列表对应的地址值没变
'''
```

### set

```python
"""
********************************************************************************************
                                        集合Set
********************************************************************************************
"""
'''
集合Set
    这里的集合Set就和java中集合Collection的子接口Set相同
    本质上都是无序，不重复的，没有下标的，用hash函数完成的，类比HashSet()

    常见的成员方法与性质
        与列表、元组相同，都可以用len(Object obj)返回列表、元组、集合的长度
        都可以用.add()   .remove()   .clear()方法
        只要与下标无关的方法都是可以用的
        
    集合的特殊性质
        集合的交集、并集、差集、对称差集
            交集  a | b
            并集  a & b
            差集  a - b
            对称差集  a ^ b(也可以按照异或理解，本身对称差集就是异或)
'''
```

### dict

```python
"""
********************************************************************************************
                                        字典dict
********************************************************************************************
"""
'''
字典dict
    注意：
        字典虽然是可变数据类型，但是是由于字典中的键值对中国的值是可变的数据，字典的键不能变，字典的键也必须是不可变数据类型，一个键值对整体
            可以在字典中添加或者删除
            eg.d = {[1, 2, 3]:"小明"}
                print(d)
            # 这个时候会报错，因为键值对中的键是可变数据类型列表，这是不允许的，因为键是身份标签，是与hash值紧密相连的，是不能变的
        字典的所有键值对中，键的名字不能重名，如果后来加入的键与之前的键相同，这个时候可能加入失败，有可能覆盖原来的键，要看具体的实现情况
    字典元素的访问与添加
        访问与修改
            d1[键名]就是访问了键名对应的hash地址值，这个时候如果print(d1[键名])就是打印这个值里面的元素，
                如果写d1[键名] = "test" 就是将这个hash地址的数据修改为了test，如果本身没有这个键名，相当于原来
                这个键名对应的hash地址值上标注的是这个位置没有数据，现在改动了值了，就相当于添加了进去，尽管之前没有这个东西
                但d1[键名] = "test"之后，就相当于在哈希表中就能找到<键名:"test">这个键值对了，实现了默认的直接添加
        删除键值对
            del d1[键名],相当于给这个hash地址值上添加标注，说明这里现在没有数据
    字典的常见方法
        同集合Set由于本质上是hash函数与hashcode所以，只要与索引值无关的方法，都与string,tuple,list相同
        d1.fromkeys(列表, Object default)  # 利用列表中的值作为键值对的键名，批量创建字典中的键值对
            键值对中的值用default值，如果没有传入default参数，默认用"None"
        查找(以字典d1为例)
            a(键名) in b(字典)
            d1(字典名)[a(键名)]就能获取键的值，但是如果根本不存在这个键，就会报错
            d1.get(Object 键名, Object default), 获取键名对应的键值，如果没有这个键名返回就default值，并添加<键名:default>进入字典
            d1.get(Object 键名)   获取键名对应的键值，如果没有这个键名就报错
            d1.setdefault(Object 键名, Object default), 与d1.get(Object 键名, Object default)相同
            d1.setdefault(Object 键名), 与d1.get(Object 键名)相同
            d1.items()  将字典中的所有键值对变成元组形式，所有的元组装在列表中，返回这个列表，即[(key1:value1), (key2:value2)]
                        # 注意是items(), 不是item(),有个s!!
            d1.keys()   将字典中所有的键名装在列表中，然后返回这个列表
            d1.values() 将字典中所有的键值装在列表中，然后返回这个列表
        改
            d1.update(dict newsrc)  # 用新字典newsrc对原来的字典d1进行更新，没有的就加入，有的但是不同的就覆盖，src中没有的就不管
        删
            d1.clear()
            d1.pop(Object 键名, Object default),  删除键名对应的键值对，并返回这个键对应的值，如果没有这个键名返回就default值
            d1.popitem()    随机删除一个键值对
        实现遍历
            for(i in d1.items()):
                print(i[0], i[1])
            
'''
```







## 函数

```python
"""

#                                          目录
#   函数
#       函数的定义
#       变量在函数内外的作用域 && global关键字
#       匿名函数lambda

"""
```

### 函数的定义

```python
"""
********************************************************************************************
                                        函数的定义
********************************************************************************************
"""
'''
1、函数的简单定义
    def 函数名（参数列表）:
        return  
    注意
        1)如果没有return,函数默认返回None
        2)python中函数的return可以返回多个参数值，只要按照参数的顺序在调用函数的时候一个一个的接收即可
            eg. def func():
                    return a, b, c
                outA, outB, outC = func()
2、函数的参数列表
    def func(参数A, 参数B, 参数C, 默认参数A=a, 默认参数B=b)
        
    位置参数
        定义的时候的参数A就可以在函数调用的时候，传入参数，进行赋值。位置参数的意思是，赋值给谁，按照调定义的时候参数的顺序进行赋值，把调用函数的时候传入的参数赋值给对应的参数
        位置参数需要子啊关键字参数之前
    关键字参数
        赋值的目标也是函数定义的时候定义的普通参数,或者默认参数，例如参数C
        通过参数名=参数值进行赋值，所以必须赋值的时候在位置参数赋值的后面，否则会打乱位置参数按照位置进行赋值的操作
    默认参数
        默认参数就是定义的时候就赋值的参数，如果没有通过关键字来赋值，就使用默认值，如果后面又利用关键字餐参数赋值了，就把原来的默认值覆盖就行了
    可变参数
        定义函数时，有时候我们不确定调用的时候会传递多少个参数(不传参也可以)。此时，可用包裹(packing)位置参数，或者包裹关键字参数，来进行参数传递，会显得非常方便。
        1、包裹位置传递
            def func(*args):
            我们传进的所有参数都会被args变量收集，它会根据传进参数的位置合并为一个元组(tuple)，args是元组类型，这就是包裹位置传递。
        2、包裹关键字传递
            def func(**kargs):
            kargs是一个字典(dict)，收集所有关键字参数
        函数中args与kargs的使用
            *args就得到了可变参数组成的元组
            **kargs就得到了可变参数的键值对组成的字典
            接下来的使用就和元组和字典的使用方法相同了
    多种参数混用的注意事项
        基本原则是：先位置参数，默认参数，包裹位置，包裹关键字(定义和调用都应遵循)
'''
```

### 变量在函数内外的作用域与global关键字

```python
"""
********************************************************************************************
                                    变量在函数内外的作用域
                                        global关键字
********************************************************************************************
"""
'''
 Python中定义函数时，若想在函数内部对函数外的变量进行操作，就需要在函数内部将其声明其为global 变量。
    添加了global关键字后，则可以在函数内部对函数外的对象进行操作了，也可以改变它的值。
    eg.
        x = 4
        def my():
            x = 8
            print("x = ", x)
        my()
        print("x = ", x)
        # 结果是：
        x = 8
        x = 4
需要注意的是 global 需要在函数内部声明，若在函数外声明，则函数依然无法操作 x 。
        x = 4
        def my():
            global x
            x = 8
            print("x = ", x)
        print("x = ", x)
        my()
        print("x = ", x)
        # 结果是：
        x = 4
        x = 8
        x = 8
global语句被用来声明 x 是全局的变量。可以使用同一个 global 语句指定多个全局变量。全局无法使用局部变量，只有对应的局部作用域可用。
         global x, y, z    # 一个global关键字定义多个全局变量
'''
```

### 匿名函数lambda

```python
"""
********************************************************************************************
                                        匿名函数lambda
********************************************************************************************
"""
# Eg.   print(lambda x,y: x+y)
```





## 多文件编写

```python
"""
**********************************************************************************************




                                        多文件编写 





**********************************************************************************************
"""
```

### 库、包、模块的区别和联系

```python
"""
********************************************************************************************
                                        库、包、模块的区别与联系
********************************************************************************************
"""
'''
    1、嵌套关系
        库
        |__包
            |__模块(源文件.py)
    2、不管是包还是库，本质上都是文件夹，所以在import的时候都差不多
        eg.torch和pillow本质上就是文件夹，这个文件夹的下面又有许多文件加，import的时候，就是给这些文件夹创建索引，能让编译器运行的时候能找到文件的实际位置
        
'''
```

### import函数

```python
"""
********************************************************************************************
                                        import函数
********************************************************************************************
"""
'''
#   常见的import方法
#       1、同一级或者下n级的时候的导入方法
#           1)import 包名、库、模块名           #   导入了这个文件夹的地址，但是没有导入代码，要用的话就要继续.eg.torch.cuda.isavailable()
#           2)from 文件夹 或 模块名 import *   #   导入了所有的文件和代码
#           3)from 文件夹.文件夹.文件夹.模块名(.py文件) import 类名、函数名、属性名(变量名)  
#               # 库、包本质上就是文件夹，import直接使用的时候就是打开文件夹，或者打开文件的操作
#                   导入的本质就是把代码直接添加到文件的最前方，导入什么就添加什么，导入了的就可以直接用，例如仅仅import pillow，就相当于把pillow这个文件夹的所有文件
#                   导入，但是要调用里面的Image.open()方法，就需要pillow.Image.open(图片地址)，而不是直接用Image.open()，因为Image在pillow中，编译器会找不到位置
#                   但是如果用from pillow import Image(这是一个类)，就相当于源文件的最开头单独把pillow.Image重命名为Image，就可以直接用Image.show()了，相当于
#                   直接把Image这个类在源文件的开头写了一遍，就可以直接用了
#       2、上n级的时候的导入方法
#           向下文件夹是      (隐藏了是在项目、源文件所在目录)文件夹A.文件夹B
#           向上找文件夹是     (隐藏了是在项目、源文件所在目录)..文件A
#           一个.向下打开文件夹，两个.是返回上一级文件夹
'''

```

### name属性

```python
"""
********************************************************************************************
                                        __name__属性
********************************************************************************************
"""
'''
    1)__name__属性是对于源文件而言的，具体来说__name__是一个变量，每个源文件都有这个变量，这个变量的值的作用域就是整个源文件
        这就好像自己在源文件开头定义了一个值__name__值一样，
    2)__name__值默认为源文件的文件名（不含后缀.py），但是如果程序是从改源文件开始运行的，源文件的__name__属性就会被改为__main__
    3)可以用if(__name__ == "__main__"):
        来区分哪些代码是这个文件作为main文件来执行的时候运行
        那些代码是被Import的时候运行，
        只要被import，那整个源文件的代码都是要被运行的，这个时候if(__name__ == "__main__"):就可以把作为main运行的代码保护起来，只有在
            以main运行的时候才能进入if判断语句
'''
```



## 其他杂项

```python
"""
**********************************************************************************************





                                   其他杂项





**********************************************************************************************
"""
'''
1)python中末尾可以打分号，也可以省略

2)a = a ** 10 两个**表示a的10次方

3)python中的逻辑运算    ||用or表示   （java）a || b     （python）a or b
                    &&用and表示         a && b              a and b
                    !用not表示           !a                  not a
4)python中的字符串运算
    s1 + s2表示字符串的拼接（中间没有其他东西）
    s1 * 3 表示将字符串s1复制3次，拼在一起，eg s1 = "abc"  那s1 * 3 = "abcabcabc"
5)python中的命名格式规范
    类名用大驼峰命名法
    函数与变量名并不使用小驼峰命名法，统一都使用小写，然后用_进行隔开单词
6)python中的运算符
    a**b表示a的b次方
    a/b表示除法，会自动转化为float类型
    a//b表示整除，同C或者java中的/
    
7)变量的删除
    del a
    就类似于free a,在重新赋值之前，都是不能用的了

8)读取变量的地址值
    id(Object obj)  # 返回obj的地址值
    python中Number型、元组、字符串都是不可变数据类型，一个地址对应一个数据的值，数据改变，就必须更改地址
    这与c或者java中int不是引用类型完全不同
    
    集合（列表）、字典都是可变数据类型，值改变，地址值不变

9)range(start, stop [,step])
    返回的是一个类似列表的range类型，基本与列表相同，但是不可更改，由于一般只拿来遍历for的时候用，所以就和匿名对象基本一个用法
    start 指的是计数起始值，默认是 0；stop 指的是计数结束值，但不包括 stop ；step 是步长，默认为 1，不可以为 0 。
    range() 方法生成一段左闭右开的整数范围。
    
10)函数如果不带return 相当于默认返回None

11)python中if判断句子中，None  ""(空字符串)  0  ()(空元组)  [](空列表)  {}(空字典)  false表示假
     逻辑运算与或非用and or not表示, 而不是&& || !
'''
```







# 机器学习

## 监督学习和无监督学习的区别

有监督学习

​		使用的是有标签的数据，相当于机器训练的任务是有正确答案的。通常用于分类和回归问题

无监督学习

​		没有标签，没有正确答案，通常用于聚类的问题，通过机器使用相关算法来实现对数据进行分类。

强化学习

​		通过机器（算法）与环境的反应结果来不断调整算法的策略。

半监督学习

​		先通过小部分人工设立的标签来训练模型，然后通过这个训练的半成品模型给更多的数据打标签，从而节省打标签的时间。最后再用这些标签（挑选可信度较高的）进行后续大规模的训练。

## 机器学习和深度学习的区别

1.  深度学习是机器学习的一种算法，所以深度学习其实是包含在机器学习中的
2.  机器学习需要人工去提取处理对象的特征，需要人为的先总结出来那些地方重要，然后通过算法的方式告诉程序，然后再利用程序进行训练，解决实际的问题。但是深度学习可以通过卷积自己获取处理对象的特征，然后通过梯度下降之类的方法不断调整自己提取特征的方法，最后实现问题解决更加精确。

## 偏导数、链式法则、梯度、矩阵等数学概念在机器学习中的作用

1.  梯度

    深度学习中，需要对神经网络中的权重参数进行修改，修改的方式就是通过梯度下降。深度学习在训练的时候会定义一个损失函数，用来描述网络预测的结果与真实的标签之间的差距，而梯度的方向是函数在某一点增加速度最快的方向，沿着梯度的反方向更改权重参数就能够实现损失函数朝着尽可能的最低点靠近，这样就能逐渐提高模型的准确度。

2.  偏导数与链式法则

    有数学知识，梯度就是函数在某点上升最快的单位向量，他的各个分量就是函数在该点对各个方向的偏导数，因此求梯度需要求偏导数

    而对于复合函数，偏导数的计算就需要利用链式法则一步一步进行拆分，最终才能变为求导公式能够解决的问题

3.  矩阵

    深度学习中，一个模型的预测结果是通过入参和权重参数相乘之后再相加，然后经过激活函数、池化等等操作得到的，采用矩阵的运算形式可以很方便的存储和计算。

## 常见的激活函数

1.  Sigmoid 函数

    $${\LARGE f(x)=\frac{1}{1+e^{-x}}}$$

    容易发生梯度消失，导致反向传播更新参数难以进行，从而参数难以更新，结果准确率较低

2.  ReLu

    $$f(x)=\left\{\begin{matrix}x, x> 0  \\0, x\le 0 \end{matrix}\right.$$

    解决了梯度消失的问题，但当输入为负数的时候，函数值为0，反向传播停止，部分神经元无法进行更新，存在“神经元死亡”的问题

3.  Tanh 函数（双曲正切函数）

    $${\LARGE \tanh(x) = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}} } $$

    Tanh 函数将输入值映射到 -1 到 1 之间，与 Sigmoid 函数类似。仍然存在梯度消失的问题，并且设计指数运算成本较高

4.  Softmax 

    $${\LARGE f(x_{i})=\frac{e^{x_{i}}}{ {\textstyle \sum_{j=1}^{n} e^{x_{j}}} } } $$

    主要用于多分类问题，将输入的数值转换为概率分布，使得输出的每个元素都在 0 到 1 之间，且所有元素的和为 1。

    (以下的激活函数暂时没用过)

5.  Leaky ReLU 函数

    $$f(x)=\left\{\begin{matrix}x, x> 0  \\\alpha x, x\le 0 \end{matrix}\right.$$

    $$\alpha$$是一个远小于 1 的正数，通常取 0.01 左右。

    1.  特点：

        在输入小于等于 0 的部分，不再是简单地输出 0，而是输出一个较小的斜率$$\alpha$$乘以输入值。

    2.  优点：

        一定程度上缓解了 ReLU 函数中 “神经元死亡” 的问题，保留了 ReLU 函数的计算高效、快速收敛等优点。

    3.  缺点：

        引入了额外的超参数，需要进行调参；函数的输出在 0 点处不连续，可能会对训练产生一定的影响。

6.  ELU 函数（指数线性单元）

    $$f(x)=\left\{\begin{matrix}x, x> 0  \\\alpha (e^x-1), x\le 0 \end{matrix}\right.$$

    1.  特点：

        在输入大于 0 的部分，输出与输入相同；输入小于等于 0 的部分，通过指数函数进行变换。

    2.  优点

        具有 ReLU 的基本优点，能够避免 “神经元死亡” 问题；输出的均值接近 0，有助于加快训练速度；在一定程度上缓解了梯度消失问题。

    3.  缺点

        计算相对复杂，涉及到指数运算，计算成本较高。

7.  ==**GELU 函数**==

    $$f(x)=xP(X \le x)=x\Phi (x)$$

    其中$$\Phi (x)$$是标准正态分布的累积分布函数。

    1.  特点：是一种基于高斯分布的激活函数，在输入值较小时，会对输入进行一定的平滑处理；在输入值较大时，接近线性变换。

    2.  优点

        在自然语言处理等任务中表现良好，能够提高模型的性能；具有较好的正则化效果，可以减少过拟合的风险。

-   神经网络的基本结构

    -   输入层

    -   隐藏层

        隐藏层可以有一层或多层，每层包含多个神经元（节点）

        -   卷积层
        -   池化层
        -   激活函数

    -   输出层

        如果是分类任务，就是每一类别的概率值，如果是回归任务，就是一个数值

    -   连接方式

        -   全连接
        -   部分链接

    ## 机器学习中的数据处理

    1.  数据收集

    2.  数据清洗

        数据清洗是去除数据中的噪声、异常值和错误数据的过程。

        常见的数据清洗的方法：

        1.  缺失值处理：

            采用填充方法，如均值填充、中位数填充、众数填充等。

            也可以使用插值法、回归法等。

        2.  异常值处理：

            采用删除、替换或修正的方法

        3.  重复值处理

    3.  数据预处理

        数据预处理是将原始数据转换为适合机器学习算法的格式和特征的过程。

        1.  归一化

            数据归一化是将数据的特征值缩放到一个特定的范围内，通常是 [0,1] 或 [-1,1]。这可以避免特征值的尺度差异对模型性能的影响。

            常见的数据归一化方法有最小 - 最大归一化、Z-score 标准化等。

        2.  标准化

            数据标准化是将数据的特征值转换为具有零均值和单位方差的分布。这可以使不同特征具有相同的尺度，便于模型的学习和优化

            Z-score 标准化是一种常见的数据标准化方法。

        3.  特征选择

            特征选择是从原始特征集中选择一部分最相关、最有代表性的特征，以减少特征维度、提高模型性能和可解释性。

            常见的特征选择方法有过滤式方法（如方差选择、相关系数法等）、包裹式方法（如递归特征消除等）和嵌入式方法（如基于正则化的方法等）。

        
