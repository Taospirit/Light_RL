# todo
## 20220320:
- 重新整理代码，确定统一结构：
  ```
  def main():
    train()
    eval()
  ```
- 更新保存模型方式，加上步数戳，默认load最新模型


# QA
## 0320:
- 无法安装运行rocket环境
  - gym更新导致rendering api更换，待check
- 无法自动补全[fixed]
  - 从setting sync同步的python解析器路径不对，需要根据实际情况修改。 



$$
  L_{total} = \alpha * L_{exp} + \beta * L_{ldm} + L_{angle}
$$