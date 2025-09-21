from pymilvus import connections, utility, Collection

# Kết nối tới Milvus host remote
connections.connect("default", host="172.18.213.122", port="19530")

# Liệt kê collection
print("Collections:", utility.list_collections())

# Kiểm tra schema
collection_name = "course_chunks"
if utility.has_collection(collection_name):
    collection = Collection(collection_name)
    for field in collection.schema.fields:
        print(f"{field.name}: {field.dtype}")

# Kiểm tra số lượng record
print("Entities count:", collection.num_entities)
