# 1. 自定义数据读取格式

```
torch.utils.data.Dataset
```

`torch.utils.data.Dataset`是代表这一数据的**抽象类**。你可以自己定义你的数据类，继承和重写这个抽象类，非常简单，只需要定义`__len__`和`__getitem__`这个两个函数。

用`torch.utils.data.Dataset`与`Dataloader`组合可以得到数据迭代器。在每次训练时，利用这个迭代器输出每一个batch数据，并能在输出时对数据进行相应的预处理或数据增广操作。

**`torch.utils.data.Dataset`与`torch.utils.data.DataLoader`的理解**

* pytorch提供了一个数据读取的方法，其由两个类构成：torch.utils.data.Dataset和DataLoader
* 我们要自定义自己数据读取的方法，就需要继承torch.utils.data.Dataset，并将其封装到DataLoader中
* torch.utils.data.Dataset表示该数据集，继承该类可以重载其中的`__len__`和`__getitem__`两个函数，实现多种数据读取及数据预处理方式
* torch.utils.data.DataLoader 封装了Data对象，实现单（多）进程迭代器输出数据集

## 1.1  `torch.utils.data.Dataset`

1. 要自定义自己的Dataset类，至少要重载两个方法，`__len__`, `__getitem__`
2. `__len__`返回的是数据集的大小
3. `__getitem__`实现索引数据集中的某一个数据
4. 除了这两个基本功能，还可以在`__getitem__`时对数据进行预处理，或者是直接在硬盘中读取数据，对于超大的数据集还可以使用`lmdb`来读取

## 1.2 `torch.utils.data.Dataloader`

1. 把自己的Dataset类作为一个参数传递给Dataloader，Dataloader将Dataset或其子类封装成一个迭代器
2. 这个迭代器可以迭代输出Dataset的内容
3. 同时可以实现多进程、shuffle、不同采样策略，数据校对等等处理过程

**当然，也可以不使用pytorch的Dataloder加载数据，而是自己实现一个加载类，graphsnn就是自己实现了一个DataReader。**