### Part 1: Repair Suggestion Extraction (基于动作)

**核心定义**：修复建议被建模为一系列 **"Action Tuples"，**(Verb,Object)

### 1. 算法描述 (Description)

To quantify the quality of repair suggestions, we extract Repair Actions (RA) based on syntactic dependency parsing. A repair action is defined as a tuple (v,o) where v is an action verb (e.g., "modify", "add") and o is the object entity affected by the action (e.g., "condition", "variable"). We utilize a dependency parser to identify verbs and traverse their children nodes to locate direct objects (dobj), indirect objects (iobj), prepositional objects (pobj), and passive subjects (nsubjpass).

### 2. 伪代码 (Pseudo-code)

`Algorithm: Extract_Repair_Actions
Input:  Suggestion_Sentence (S)
Output: Set of Repair Actions (RA_Set)

# 1. 定义我们关心的宾语依存关系标签
TARGET_OBJ_DEPS = {'dobj', 'iobj', 'pobj', 'nsubjpass'}

Initialize RA_Set = empty_set

# 2. 解析句子生成依存树
Doc = NLP_Parser(S) 

For each Token t in Doc:
    # 步骤 A: 寻找核心动词
    If t.POS == 'VERB':
        Action_Verb = t.lemma_  # 取词干 (e.g., "added" -> "add")
        Target_Object = NULL
        
        # 步骤 B: 遍历该动词的子节点寻找宾语
        For each Child c in t.children:
            
            # 情况 1: 直接/间接/被动宾语
            If c.dependency_label in TARGET_OBJ_DEPS:
                Target_Object = c.text
                
            # 情况 2: 介词宾语 (例如 "rely on configuration")
            # 结构: rely(Verb) -> on(Prep) -> configuration(pobj)
            Else If c.dependency_label == 'prep':
                For each GrandChild gc in c.children:
                    If gc.dependency_label == 'pobj':
                        Target_Object = gc.text
                        
        # 步骤 C: 如果找到成对的动作和对象，存入集合
        If Target_Object is not NULL:
            RA_Set.add( (Action_Verb, Target_Object) )

Return RA_Set`

---

### Part 2: Root Cause Extraction (基于语义短语)

**核心定义**：根本原因被建模为 **"Defect Concepts" (缺陷概念)**，通常表现为特定的名词短语结构。

### 1. 算法描述 (Description)

To evaluate the accuracy of root cause analysis, we extract Key Defect Concepts (KDC). Unlike repair actions, root causes are typically described as states or specific technical terms. We design a three-rule extraction strategy based on noun phrases:

1. **Compound Nouns:** Captures technical terms formed by multiple nouns (e.g., "buffer overflow").
2. **Adjectival Modifiers:** Captures defects described by state adjectives (e.g., "infinite loop", "null pointer").
3. **Negative Constraints:** Explicitly captures missing or absent components using negative determinants (e.g., "missing check", "no validation"), which are critical in vulnerability analysis.

### 2. 伪代码 (Pseudo-code)

`Algorithm: Extract_Root_Cause_Concepts
Input:  Root_Cause_Sentence (S)
Output: Set of Defect Concepts (RC_Set)

# 定义否定相关的关键词 (用于规则 3)
NEGATIVE_TERMS = {'missing', 'no', 'lack', 'without'}

Initialize RC_Set = empty_set
Doc = NLP_Parser(S)

For each Token t in Doc:
    # 核心逻辑: Root Cause 的核心通常是一个名词 (Head Noun)
    If t.POS == 'NOUN':
        Head_Noun = t.text
        Modifier = NULL
        
        # 遍历修饰这个名词的子节点
        For each Child c in t.children:
            
            # Rule 1: Compound Nouns (名词+名词)
            # Example: "Buffer(c) overflow(t)"
            If c.dependency_label == 'compound':
                Modifier = c.text
                
            # Rule 2: Adjectival Modifiers (形容词+名词)
            # Example: "Null(c) pointer(t)"
            Else If c.dependency_label == 'amod':
                Modifier = c.text
            
            # Rule 3: Negative Constraints (否定词+名词)
            # Example: "No(c) validation(t)", "Missing(c) check(t)"
            # 注意: 'missing' 经常被标记为 amod，但在某些 parser 可能是 verb(acl)
            # 这里我们通过词表匹配来强化提取
            If c.text.lower() in NEGATIVE_TERMS:
                 Modifier = c.text

            # 如果找到修饰语，组合成短语
            If Modifier is not NULL:
                Phrase = Modifier + " " + Head_Noun
                RC_Set.add(Phrase)

Return RC_Set`