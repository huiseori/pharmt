# import chromadb
# from chromadb.utils import embedding_functions
# import openai
#
# # OpenAI API 키 설정 (환경 변수에서 가져오는 것이 좋습니다)
# from pharmtracker import settings
# openai.api_key= settings.OPENAI_API_KEY
#
# # OpenAI 임베딩 함수 설정
# openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#     api_key=openai.api_key,
#     model_name="text-embedding-3-small"
# )
#
# # ChromaDB 클라이언트 설정
# client = chromadb.Client()
# collection = client.create_collection(name="test_drug_embeddings", embedding_function=openai_ef)
#
# # 테스트 데이터 삽입
# test_data = [
#     {"id": "1", "name": "아스피린", "description": "해열, 진통제로 사용되는 약물입니다."},
#     {"id": "2", "name": "이부프로펜", "description": "소염진통제로 사용되는 약물입니다."},
#     {"id": "3", "name": "파라세타몰", "description": "해열, 진통제로 사용되는 약물입니다."}
# ]
#
# collection.add(
#     ids=[item["id"] for item in test_data],
#     documents=[f"{item['name']}: {item['description']}" for item in test_data],
#     metadatas=test_data
# )
#
# # 데이터 쿼리
# query = "진통제"
# results = collection.query(
#     query_texts=[query],
#     n_results=2
# )
#
# print(f"Query: {query}")
# print("Results:")
# for id, document, metadata in zip(results['ids'][0], results['documents'][0], results['metadatas'][0]):
#     print(f"ID: {id}")
#     print(f"Document: {document}")
#     print(f"Metadata: {metadata}")
#     print("---")
#
# # 컬렉션 삭제 (테스트 후 정리)
# client.delete_collection("test_drug_embeddings")