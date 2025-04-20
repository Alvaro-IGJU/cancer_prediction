import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import networkx as nx

# Cargar los datasets
df_historial = pd.read_csv("data/historial_medico.csv")
df_cancer = pd.read_csv("data/analisis_cancer.csv")
df_imagenes = pd.read_csv("data/historial_medico_imagenes.csv")

# Crear/conectar a la base de datos
conn = sqlite3.connect("prediccion_cancer.db")
cursor = conn.cursor()

# Crear tabla pacientes
cursor.execute("""
CREATE TABLE IF NOT EXISTS pacientes (
    id INTEGER PRIMARY KEY,
    Sexo TEXT,
    Age INTEGER,
    "Family history" TEXT,
    smoke TEXT,
    alcohol TEXT,
    obesity TEXT,
    diet TEXT,
    Screening_History TEXT,
    Healthcare_Access TEXT,
    Survival_Prediction TEXT
)
""")

# Crear tabla analisis_cancer
cursor.execute("""
CREATE TABLE IF NOT EXISTS analisis_cancer (
    id INTEGER PRIMARY KEY,
    cancer_stage TEXT,
    tumor_size INTEGER,
    early_detection TEXT,
    inflammatory_bowel_disease TEXT,
    relapse TEXT
)
""")

# Crear tabla imagenes
cursor.execute("""
CREATE TABLE IF NOT EXISTS imagenes (
    id INTEGER PRIMARY KEY,
    imagename TEXT
)
""")

# Insertar los datos desde los DataFrames
df_historial.to_sql("pacientes", conn, if_exists="replace", index=False)
df_cancer.to_sql("analisis_cancer", conn, if_exists="replace", index=False)
df_imagenes.to_sql("imagenes", conn, if_exists="replace", index=False)

# Confirmar y cerrar
conn.commit()
conn.close()

print("✅ Base de datos creada correctamente con 3 tablas relacionadas por 'id'.")

# ============ DIAGRAMA TIPO ERD EN PNG ============

cols_pacientes = df_historial.columns.tolist()
cols_analisis = df_cancer.columns.tolist()
cols_imagenes = df_imagenes.columns.tolist()

G = nx.DiGraph()

# Nodos tabla
G.add_node("pacientes", shape="box")
G.add_node("analisis_cancer", shape="box")
G.add_node("imagenes", shape="box")

# Columnas como subnodos
for col in cols_pacientes:
    G.add_node(f"pacientes.{col}")
    G.add_edge("pacientes", f"pacientes.{col}")
for col in cols_analisis:
    G.add_node(f"analisis_cancer.{col}")
    G.add_edge("analisis_cancer", f"analisis_cancer.{col}")
for col in cols_imagenes:
    G.add_node(f"imagenes.{col}")
    G.add_edge("imagenes", f"imagenes.{col}")

# Relaciones
G.add_edge("pacientes", "analisis_cancer", label="id")
G.add_edge("pacientes", "imagenes", label="id")

# Dibujar y guardar
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, seed=42, k=0.5)
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=8, font_weight="bold", edge_color="gray")
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
plt.title("Esquema de la base de datos: Tablas, columnas y relaciones", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.savefig("esquema_base_datos.png")

print("🖼️ Diagrama guardado como 'esquema_base_datos.png'")

# ============ ARCHIVO .DBML PARA DIAGRAMA VISUAL ============

with open("esquema.dbml", "w") as f:
    f.write("""
Table pacientes {
  id INTEGER [pk]
  Sexo TEXT
  Age INTEGER
  Family_history TEXT
  smoke TEXT
  alcohol TEXT
  obesity TEXT
  diet TEXT
  Screening_History TEXT
  Healthcare_Access TEXT
  Survival_Prediction TEXT
}

Table analisis_cancer {
  id INTEGER [pk, ref: > pacientes.id]
  cancer_stage TEXT
  tumor_size INTEGER
  early_detection TEXT
  inflammatory_bowel_disease TEXT
  relapse TEXT
}

Table imagenes {
  id INTEGER [pk, ref: > pacientes.id]
  imagename TEXT
}
""")

print("📄 Archivo 'esquema.dbml' generado para usar en dbdiagram.io")
