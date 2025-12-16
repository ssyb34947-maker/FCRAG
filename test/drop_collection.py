import argparse
from pymilvus import connections, utility

def drop_collection(collection_name):
    """删除指定的Milvus集合"""
    # 连接到Milvus
    connections.connect(
        alias="default",
        host="127.0.0.1",
        port="19530"
    )
    
    print(f"已连接到Milvus")
    
    # 检查集合是否存在
    if utility.has_collection(collection_name):
        # 删除集合
        utility.drop_collection(collection_name)
        print(f"集合 '{collection_name}' 已成功删除")
    else:
        print(f"集合 '{collection_name}' 不存在")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="删除Milvus集合")
    parser.add_argument("--name", required=True, help="要删除的集合名称")
    
    args = parser.parse_args()
    
    drop_collection(args.name)