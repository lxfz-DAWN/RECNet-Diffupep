# NetRobustTest

This is a test project for NetRobust.
we will choose some peptides which are known to be functionally important and test their ability to be predicted by NetRobust.

- **检测目标**：ESM网络打分与pydock打分基本一致，同时对于泛化的蛋白质效果相仿，即可认为网络的泛化能力较强，鲁棒性好。
- **选择的蛋白种类**：
  - [ ] 不同性质的蛋白：强电负性、
  - [ ] 不同种的蛋白：已有病毒序列的多肽：
    - [ ] 原有序列的变体：SARS、SARS-CoV-2、MERS、OC43、
    - [ ] 同样拥有6-HB结构：
    - [ ] 同属呼吸道病毒丨I型胞膜病毒
    - [ ] 乱七八糟的病毒
  - 随机序列比对
    - [x] 纯序列比对
    - [x] 无义序列比对
    - [x] 随机序列比对