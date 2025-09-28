class RetrievalRouter:
    """
    Orchestrates the retrieval process by routing queries based on their
    analyzed type.
    """
    def __init__(self, bm25_searcher, chroma_collection, funds_df, query_generator):
        self.bm25_searcher = bm25_searcher
        self.chroma_collection = chroma_collection
        self.funds_df = funds_df
        self.query_generator = query_generator
        self.all_fund_names = funds_df['fund_name'].tolist()

    def retrieve(self, analyzed_query: dict, search_mode: str = 'hybrid') -> list:
        query_type = analyzed_query.get('query_type', 'ambiguous')
        intent = analyzed_query.get('intent', '')
        results = []
        print(f"Routing query type '{query_type}' with search mode '{search_mode}'...")

        if query_type in ["definition", "entity_lookup", "ambiguous"]:
            if search_mode == 'semantic':
                chroma_res_raw = self.chroma_collection.query(query_texts=[intent], n_results=3, include=['documents', 'metadatas'])
                if chroma_res_raw and chroma_res_raw['documents']:
                    docs = chroma_res_raw['documents'][0]
                    metas = chroma_res_raw['metadatas'][0]
                    results = [{'document': doc, 'metadata': meta} for doc, meta in zip(docs, metas)]
            elif search_mode == 'lexical':
                results = self.bm25_searcher.search(intent, n_results=3)
            else: # 'hybrid'
                bm25_res = self.bm25_searcher.search(intent, n_results=2)
                chroma_res_raw = self.chroma_collection.query(query_texts=[intent], n_results=2, include=['documents', 'metadatas'])
                chroma_res = []
                if chroma_res_raw and chroma_res_raw['documents']:
                    c_docs = chroma_res_raw['documents'][0]
                    c_metas = chroma_res_raw['metadatas'][0]
                    chroma_res = [{'document': doc, 'metadata': meta} for doc, meta in zip(c_docs, c_metas)]

                combined_results = bm25_res + chroma_res
                seen_docs = set()
                results = []
                for item in combined_results:
                    if item['document'] not in seen_docs:
                        seen_docs.add(item['document'])
                        results.append(item)

        elif query_type in ["ranking", "comparison", "hybrid_analytical"]:
            generated_code = self.query_generator.generate_query(intent, self.funds_df)
            if "ERROR" in generated_code:
                results.append({"document": generated_code, "metadata": {"source": "pandas_generator_error"}})
            else:
                try:
                    df = self.funds_df
                    query_result_df = eval(generated_code)
                    if not query_result_df.empty:
                        result_str = "Here are the results from your query:\n" + query_result_df.to_string()
                        results.append({"document": result_str, "metadata": {"source": "structured_query"}})
                    else:
                        results.append({"document": "Your query returned no results.", "metadata": {"source": "structured_query"}})
                except Exception as e:
                    error_msg = f"Failed to execute generated pandas code: {e}"
                    results.append({"document": error_msg, "metadata": {"source": "pandas_execution_error"}})
        return results