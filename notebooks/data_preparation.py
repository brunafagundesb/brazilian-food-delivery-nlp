import pandas as pd 

#Carregar o dataset original
df = pd.read_csv("data/raw/olist_order_reviews_dataset.csv")

print("Dataset original:")
print(f"Numero de linhas: {len(df)}")

#Remover os Reviews IDs duplicados
df = df.drop_duplicates(subset="review_id", keep="first")
print("\nDataset após remover duplicatas:")
print(f"Numero de linhas: {len(df)}")

#Combinar review title e review message
df["review_comment_title"] = df["review_comment_title"].fillna("")
df["review_comment_message"] = df["review_comment_message"].fillna("")

df["review_text"] = (
    df["review_comment_title"].str.strip() + " " 
    + df["review_comment_message"].str.strip()
).str.strip()

#Manter apenas reviews com texto
df =df[df["review_text"] != ""].copy()

print("\n Reviews com texto:")
print(f"Numero de linhas: {len(df)}")

#Criar rótulos de sentimento
sentiment_mapping = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive"
}

df["Sentiment"] = df["review_score"].map(sentiment_mapping)

#Checar distribuição de sentimentos
print("\nDistribuição de sentimentos:")
print(df["Sentiment"].value_counts())

#Salvar o dataset preparado
df.to_csv("data/processed/reviews_cleaned.csv", index=False, encoding="utf-8")

print("\nDataset preparado salvo!")

