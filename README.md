# data_processing

### Build venv

```powershell
./build_venv.ps1
```

### Import

```python
import data_processing as dp
dp.init(__file__)
```

`dp.init()` is used to relatively locate to the script file

### Read csv

```python
train_dataframe = dp.read_csv("train")
test_dataframe = dp.read_csv("test")
```

### Analysis

```python
dp.analysis(train_dataframe)
```

```powershell
Columns:
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
Number of row: 891

PassengerId     is unique
Survived        is not unique, "0" repeated 549 times
Pclass          is not unique, "3" repeated 491 times
Name            is unique
Sex             is not unique, "male" repeated 577 times
Age             is not unique, "24.0" repeated 30 times
SibSp           is not unique, "0" repeated 608 times
Parch           is not unique, "0" repeated 678 times
Ticket          is not unique, "347082" repeated 7 times
Fare            is not unique, "8.05" repeated 43 times
Cabin           is not unique, "B96 B98" repeated 4 times
Embarked        is not unique, "S" repeated 644 times

Age             have 177 missing value
Cabin           have 687 missing value
Embarked        have 2 missing value
```

### Data processing

```python
dp.transform(
    train_dataframe,
    ('PassengerId', dp.Drop()),
    ('Name',        dp.Apply(len)),
    ('Sex',         dp.VocabEncode()),
    ('Age',         dp.FillNa(-1)),
    ('Ticket',      dp.VocabEncode()),
    ('Cabin',       dp.VocabEncode()),
    ('Embarked',    dp.VocabEncode())
)
dp.transformAll(train_dataframe, dp.Apply(float))
print(train_dataframe)
```

```
PassengerId -> [Drop]

Name -> [Apply]

Sex -> [VocabEncode]
Vocab: male, female

Age -> [FillNa]

Ticket -> [VocabEncode]
Vocab: 1601, 347082, CA. 2343, 3101295, 347088, CA 2144, 382652, S.O.C. 14879, 113760, 113781, 17421, 19950, 2666, 347077, 349909, 4133, LINE, PC 17757, W./C. 6608, ...

Cabin -> [VocabEncode]
Vocab: nan, B96 B98, C23 C25 C27, G6, C22 C26, D, E101, F2, F33, B18, B20, B22, B28, B35, B49, B5, B51 B53 B55, B57 B59 B63 B66, B58 B60, B77, C123, C124, C125, ...

Embarked -> [VocabEncode]
Vocab: S, C, Q, nan
```

```python
     Survived  Pclass  Name  Sex   Age  SibSp  Parch  Ticket     Fare  Cabin  Embarked
0         0.0     3.0  23.0  0.0  22.0    1.0    0.0   559.0   7.2500    0.0       0.0
1         1.0     1.0  51.0  1.0  38.0    1.0    0.0   613.0  71.2833  104.0       1.0
2         1.0     3.0  22.0  1.0  26.0    0.0    0.0   672.0   7.9250    0.0       0.0
3         1.0     1.0  44.0  1.0  35.0    1.0    0.0    47.0  53.1000   20.0       0.0
4         0.0     3.0  24.0  0.0  35.0    0.0    0.0   515.0   8.0500    0.0       0.0
..        ...     ...   ...  ...   ...    ...    ...     ...      ...    ...       ...
886       0.0     2.0  21.0  0.0  27.0    0.0    0.0   205.0  13.0000    0.0       0.0
887       1.0     1.0  28.0  1.0  19.0    0.0    0.0   144.0  30.0000   72.0       0.0
888       0.0     3.0  40.0  1.0  -1.0    1.0    2.0   132.0  23.4500    0.0       0.0
889       1.0     1.0  21.0  0.0  26.0    0.0    0.0   138.0  30.0000   91.0       1.0
890       0.0     3.0  19.0  0.0  32.0    0.0    0.0   510.0   7.7500    0.0       2.0

[891 rows x 11 columns]
```

### Tensor

```python
print(dp.toTensors(train_dataframe))
```

```python
tensor([[  0.0000,   3.0000,  23.0000,  ...,   7.2500,   0.0000,   0.0000],
        [  1.0000,   1.0000,  51.0000,  ...,  71.2833, 104.0000,   1.0000],
        [  1.0000,   3.0000,  22.0000,  ...,   7.9250,   0.0000,   0.0000],
        ...,
        [  0.0000,   3.0000,  40.0000,  ...,  23.4500,   0.0000,   0.0000],
        [  1.0000,   1.0000,  21.0000,  ...,  30.0000,  91.0000,   1.0000],
        [  0.0000,   3.0000,  19.0000,  ...,   7.7500,   0.0000,   2.0000]])
```

### Save

```python
dp.save_csv(train_dataframe, 'processed/train')
```