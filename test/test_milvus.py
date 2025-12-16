from pymilvus import connections, utility, Collection

connections.connect(
    alias="default",
    host="127.0.0.1",
    port="19530"
)

print("Milvus connected successfully")

# 查询所有数据表
collections = utility.list_collections()
print(f"当前数据库中的数据表:")
if collections:
    for collection in collections:
        print(f"  - {collection}")
else:
    print("  没有找到任何数据表")

# 显示每个表的基本信息
print("\n数据表详细信息:")
for collection_name in collections:
    try:
        # 检查表是否存在
        if utility.has_collection(collection_name):
            # 获取表的统计信息
            collection = Collection(collection_name)
            collection.load()  # 加载表以获取准确的信息
            num_entities = collection.num_entities
            print(f"  - {collection_name}: {num_entities} 条记录")
            
            # 显示表结构
            schema = collection.schema
            print(f"    结构信息:")
            print(f"      描述: {schema.description}")
            print(f"      字段:")
            for field in schema.fields:
                print(f"        - {field.name} ({field.dtype}):")
                if field.is_primary:
                    print(f"            主键: 是")
                if hasattr(field, 'dim') and field.dim:
                    print(f"            维度: {field.dim}")
                if hasattr(field, 'max_length') and field.max_length:
                    print(f"            最大长度: {field.max_length}")
        else:
            print(f"  - {collection_name}: 表不存在")
    except Exception as e:
        print(f"  - {collection_name}: 获取信息时出错 ({e})")