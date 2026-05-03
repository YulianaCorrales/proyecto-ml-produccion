"""
train.py - Script para entrenar/reentrenar el modelo del Titanic

Uso:
    python train.py <training_file.csv>

Ejemplo:
    python train.py data/train.csv
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn import __version__ as sklearn_version


def train(training_file):
    """
    Entrena un modelo RandomForest para predecir supervivencia en el Titanic
    """
    
    print("=" * 70)
    print("ENTRENAMIENTO DEL MODELO TITANIC")
    print("=" * 70)
    
    try:
        train_df = pd.read_csv(training_file)
        print(f"\n✅ Datos cargados: {training_file}")
        print(f"   Dimensiones: {train_df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo: {training_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error al leer el CSV: {e}")
        sys.exit(1)
    
    required_columns = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
    missing_columns = [col for col in required_columns if col not in train_df.columns]
    
    if missing_columns:
        print(f"\n❌ Error: Faltan las siguientes columnas en el CSV:")
        for col in missing_columns:
            print(f"   - {col}")
        sys.exit(1)
    
    categorical_features = ["Sex", "Embarked"]
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    
    X = train_df[categorical_features + numeric_features].copy()
    y = train_df["Survived"].copy()
    
    print(f"\n📊 DATOS PREPARADOS:")
    print(f"   Features (X): {X.shape}")
    print(f"   Target (y): {y.shape}")
    print(f"   Clases en target: {y.unique()}")
    print(f"   Distribución de clases:")
    print(f"      - No sobrevivió (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"      - Sobrevivió (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"\n🔀 DIVISIÓN DE DATOS:")
    print(f"   Train: {X_train.shape}")
    print(f"   Validación: {X_val.shape}")
    
    print(f"\n⚙️  CREANDO PIPELINE DE PREPROCESAMIENTO...")
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    encoder_kwargs = {"handle_unknown": "ignore"}
    if tuple(map(int, sklearn_version.split('.')[:2])) < (1, 2):
        encoder_kwargs["sparse"] = False
    else:
        encoder_kwargs["sparse_output"] = False

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**encoder_kwargs))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    print(f"   ✓ Numéricos: Imputación (mediana) + Escalamiento")
    print(f"   ✓ Categóricos: Imputación (moda) + One-Hot Encoding")
    
    print(f"\n🔨 CREANDO PIPELINE CON MODELO...")
    
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=15,
            min_samples_split=5
        ))
    ])
    
    print(f"   Modelo: RandomForestClassifier")
    print(f"   - n_estimators: 100")
    print(f"   - max_depth: 15")
    print(f"   - min_samples_split: 5")
    
    print(f"\n🚀 ENTRENANDO MODELO...")
    try:
        clf.fit(X_train, y_train)
        print(f"   ✅ Modelo entrenado correctamente")
    except Exception as e:
        print(f"   ❌ Error durante el entrenamiento: {e}")
        sys.exit(1)
    
    print(f"\n📈 EVALUACIÓN EN SET DE VALIDACIÓN:")
    
    y_val_pred = clf.predict(X_val)
    y_val_pred_proba = clf.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, y_val_pred)
    auc_score = roc_auc_score(y_val, y_val_pred_proba)
    
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   AUC ROC: {auc_score:.4f}")
    
    print(f"\n   Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=["No Sobrevivió", "Sobrevivió"]))
    
    print(f"\n🔄 CROSS-VALIDATION (5-Fold):")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    
    print(f"   Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"   Promedio: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    print(f"\n💾 GUARDANDO MODELO...")
    
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/titanic_model.pkl'
    try:
        joblib.dump(clf, model_path)
        print(f"   ✅ Modelo guardado en: {model_path}")
    except Exception as e:
        print(f"   ❌ Error al guardar: {e}")
        sys.exit(1)
    
    print(f"\n" + "=" * 70)
    print("RESUMEN DEL ENTRENAMIENTO")
    print("=" * 70)
    print(f"\n✅ Modelo entrenado exitosamente")
    print(f"   Accuracy en validación: {accuracy:.4f}")
    print(f"   AUC ROC: {auc_score:.4f}")
    print(f"   Cross-Validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"\n✅ Archivo guardado: {model_path}")
    print(f"\nPróximos pasos:")
    print(f"   1. Usa predict.py para hacer predicciones:")
    print(f"      python predict.py data/test.csv output/predictions.csv")
    print("=" * 70)


if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("=" * 70)
        print("USO: python train.py <training_file.csv>")
        print("=" * 70)
        print("\nEjemplo:")
        print("  python train.py data/train.csv")
        print("\nRequisitos del CSV:")
        print("  - Survived (target: 0 o 1)")
        print("  - Pclass, Age, SibSp, Parch, Fare (numéricos)")
        print("  - Sex, Embarked (categóricos)")
        print("\nSalida:")
        print("  - Modelo guardado en: models/titanic_model.pkl")
        print("=" * 70)
        sys.exit(1)
    
    training_file = sys.argv[1]
    train(training_file)