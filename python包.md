#  1.argparse 模块

## 1.简介

`argparse`是python用于解析命令行参数和选项的标准模块，`argparse`模块的作用是用于解析命令行参数。

## 2.基本使用

```python
# 导入模块
import argparse
# 创建对象
parser = argparse.ArgumentParser()
# 添加命令行参数
parser.add_argument()
# 调用parse_args()方法进行解析，解析之后就可以使用对象了
parser.parse_args()
```

## 3.创建解析器对象ArgumentParser

```python
ArgumentParser(prog=None, 
               usage=None,
               description=None, 
               epilog=None, 
               parents=[],
               formatter_class=argparse.HelpFormatter, 
               prefix_chars='-',
               fromfile_prefix_chars=None,         					   argument_default=None,
               conflict_handler='error', 
               add_help=True)

```

## 4. add_argument()方法

add_argument()方法用来指定程序需要接受的命令参数。

```python
add_argument(self,
             *name_or_flags: str,
             action: str | Type[Action] = ...,
             nargs: int | str = ...,
             const: Any = ...,
             default: Any = ...,
             type: (str) -> _T | (str) -> _T | FileType = ...,
             choices: Iterable[_T] = ...,
             required: bool = ...,
             help: str | None = ...,
             metavar: str | Tuple[str, ...] | None = ...,
             dest: str | None = ...,
             version: str = ...,
             **kwargs: Any) -> Action
```

* **name or flags**：参数有两种，可选参数和位置参数。parse_args()运行时，会用'-'来认证可选参数，剩下的即为位置参数。位置参数必选，可选参数可选。
* **default**：设置参数的默认值
* **help**：参数命令的介绍

## 5.parse_args()

parse_args()用于解析参数。

在大多数情况下，这意味着一个简单的`Namespace`对象将由从命令行解析出来的属性建立起来。

```python
Namespace(batch_size=64, batchnorm_dim=64, dataset='MUTAG', device='cpu', dropout_1=0.5, dropout_2=0.6, epochs=500, hidden_dim=64, log_interval=10, lr=0.009, n_folds=10, n_layers=2, seed=111, threads=0, wdecay=0.009)
```

这样，就可以通过调用ArgumentParser对象获得想要的参数。

# 2.copy模块

copy模块用于对象的拷贝操作。该模块只提供了两个主要的方法：copy.copy()与copy.deepcopy()，分别表示浅复制与深复制。

浅拷贝和深拷贝的不同仅仅是对组合对象来说，所谓的组合对象就是包含了其它对象的对象。

## 1.copy.copy()

浅拷贝，拷贝父对象，不会拷贝对象的内部的子对象。即浅复制只复制对象本身，没有复制该对象所引用的对象。

## 2.copy.deepcopy()

深拷贝，完全拷贝了父对象及其子对象。即创建一个新的组合对象，同时递归地拷贝所有子对象，新的组合对象与原对象没有任何关联。虽然实际上会共享不可变的子对象，但不影响它们的相互独立性。