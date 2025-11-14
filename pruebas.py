import create_database as BDD

DB_DIR = "chroma"
embedding_model = BDD.get_embedding_model()
db = BDD.get_db()

#+Pruebas
query_text = "Como afectan las nuevas tecnologias a Telefonica"
results = db._similarity_search_with_relevance_scores(query_text, k=1)

"""""
#Print con esteroides (muestra las mejores coincidencias) 
print("\n📘 RESULTADOS DE BÚSQUEDA 📘")
print(f"Consulta: {query_text}\n{'='*80}")

for i, (doc, score) in enumerate(results, start=1):
    content_preview = doc.page_content.strip().replace("\n", " ")
    if len(content_preview) > 300:
        content_preview = content_preview[:300] + "..."
    
    print(f"\n🔹 Resultado {i}")
    print(f"🔸 Relevancia: {score:.3f}")
    print(f"📄 Contenido: {content_preview}")
    print("-" * 80)
"""

#Muestra el mejor
results = sorted(results, key=lambda x: x[1], reverse=True)

# Tomar el más relevante
top_doc, top_score = results[0]

# Mostrar de forma limpia
print("\n📘 RESULTADO MÁS RELEVANTE 📘")
print(f"Consulta: {query_text}")
print("="*80)
print(f"🔸 Relevancia: {top_score:.3f}")

content_preview = top_doc.page_content.strip().replace("\n", " ")
print(f"\n📄 Contenido:\n{content_preview}")
print("="*80) 