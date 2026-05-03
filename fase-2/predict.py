"""
predict.py - Script para realizar predicciones con el modelo del Titanic

Uso:
    python predict.py <input_file.csv> <output_file.csv>

Ejemplo:
    python predict.py data/test.csv output/predictions.csv
"""

import pandas as pd
import joblib
import sys
import os

def predict(input_file, output_file):
    """
    Lee un CSV con datos del Titanic y genera predicciones de supervivencia
    """
    
    model_path = 'models/titanic_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Error: No se encontró el modelo en: {model_path}")
        sys.exit(1)
    
    try:
        model = joblib.load(model_path)
        print(f"✅ Modelo cargado desde: {model_path}")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        sys.exit(1)
    
    try:
        df_input = pd.read_csv(input_file)
        print(f"✅ Datos de entrada cargados: {input_file}")
        print(f"   Dimensiones: {df_input.shape}")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error al leer el CSV: {e}")
        sys.exit(1)
    
    required_columns = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
    missing_columns = [col for col in required_columns if col not in df_input.columns]
    
    if missing_columns:
        print(f"❌ Error: Faltan las siguientes columnas en el CSV:")
        for col in missing_columns:
            print(f"   - {col}")
        sys.exit(1)
    
    categorical_features = ["Sex", "Embarked"]
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    
    X = df_input[categorical_features + numeric_features].copy()
    
    passengerId = None
    if "PassengerId" in df_input.columns:
        passengerId = df_input["PassengerId"].values
    
    try:
        print("\n🔮 Realizando predicciones...")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        print(f"✅ Predicciones completadas: {len(predictions)} muestras")
    except Exception as e:
        print(f"❌ Error durante la predicción: {e}")
        sys.exit(1)
    
    results = pd.DataFrame({
        'Survived': predictions.astype(int),
        'Survival_Probability': probabilities
    })
    
    if passengerId is not None:
        results.insert(0, 'PassengerId', passengerId)
    
    try:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        results.to_csv(output_file, index=False)
        print(f"\n💾 Resultados guardados en: {output_file}")
    except Exception as e:
        print(f"❌ Error al guardar resultados: {e}")
        sys.exit(1)
    
    print(f"\n📊 RESUMEN DE PREDICCIONES:")
    print(f"   Total de registros: {len(results)}")
    print(f"   Predicciones de sobrevivencia (Survived=1): {(predictions == 1).sum()}")
    print(f"   Predicciones de no sobrevivencia (Survived=0): {(predictions == 0).sum()}")
    print(f"   Probabilidad promedio de sobrevivencia: {probabilities.mean():.4f}")
    print(f"\n✅ Predicción completada exitosamente")


if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("=" * 70)
        print("USO: python predict.py <input_file.csv> <output_file.csv>")
        print("=" * 70)
        print("\nEjemplos:")
        print("  python predict.py data/test.csv output/predictions.csv")
        print("\nRequisitos del CSV de entrada:")
        print("  - Pclass, Age, SibSp, Parch, Fare (columnas numéricas)")
        print("  - Sex, Embarked (columnas categóricas)")
        print("  - PassengerId (opcional)")
        print("\nSalida:")
        print("  - CSV con columnas: PassengerId, Survived, Survival_Probability")
        print("=" * 70)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    predict(input_file, output_file)