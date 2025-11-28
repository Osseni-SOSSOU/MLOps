import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
st.set_page_config(
    page_title="Prédicteur de Churn",
    page_icon="📊",
    layout="wide"
)

class ChurnPredictor:
    def __init__(self):
        try:
            model_data = joblib.load('churn_predictor_model.joblib')
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            st.sidebar.success("✅ Modèle chargé")
        except Exception as e:
            st.sidebar.error(f"❌ Erreur: {e}")
            st.stop()
    
    def feature_engineering(self, customer_data):
        """Recréer les features d'ingénierie"""
        data = customer_data.copy()
        
        # TenureGroup - exactement comme pendant l'entraînement
        tenure = data['tenure']
        if tenure <= 12:
            data['TenureGroup'] = '0-1'
        elif tenure <= 24:
            data['TenureGroup'] = '1-2'
        elif tenure <= 36:
            data['TenureGroup'] = '2-3'
        elif tenure <= 48:
            data['TenureGroup'] = '3-4'
        elif tenure <= 60:
            data['TenureGroup'] = '4-5'
        else:
            data['TenureGroup'] = '5-6'
        
        # ChargeToTenureRatio
        data['ChargeToTenureRatio'] = data['MonthlyCharges'] / (data['tenure'] + 1)
        
        # TotalMonthlyRatio
        data['TotalMonthlyRatio'] = data['TotalCharges'] / (data['MonthlyCharges'] + 1)
        
        # Gérer les divisions par zéro
        data['ChargeToTenureRatio'] = np.nan_to_num(data['ChargeToTenureRatio'], nan=0.0, posinf=0.0, neginf=0.0)
        data['TotalMonthlyRatio'] = np.nan_to_num(data['TotalMonthlyRatio'], nan=0.0, posinf=0.0, neginf=0.0)
        
        return data
    
    def preprocess_data(self, customer_data):
        """Prétraiter les données avec feature engineering"""
        # Appliquer le feature engineering
        engineered_data = self.feature_engineering(customer_data)
        
        # Créer DataFrame
        df = pd.DataFrame([engineered_data])
        
        # Encoder les variables catégorielles
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                try:
                    # Vérifier si la valeur existe dans l'encodeur
                    if engineered_data[col] in encoder.classes_:
                        df[col] = encoder.transform([engineered_data[col]])[0]
                    else:
                        # Utiliser la classe la plus fréquente comme fallback
                        df[col] = encoder.transform([encoder.classes_[0]])[0]
                        st.warning(f"Valeur '{engineered_data[col]}' non reconnue pour {col}, utilisation de '{encoder.classes_[0]}'")
                except Exception as e:
                    st.error(f"Erreur encodage {col}: {e}")
                    return None
        
        # Standardiser les features numériques
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        # Vérifier et réorganiser les colonnes
        missing_features = set(self.feature_names) - set(df.columns)
        extra_features = set(df.columns) - set(self.feature_names)
        
        if missing_features:
            st.error(f"Features manquantes: {missing_features}")
            return None
        
        if extra_features:
            st.warning(f"Features supplémentaires ignorées: {extra_features}")
        
        # Réorganiser selon l'ordre d'entraînement
        df = df[self.feature_names]
        
        return df
    
    def predict(self, customer_data):
        """Faire une prédiction"""
        try:
            processed_data = self.preprocess_data(customer_data)
            if processed_data is None:
                return None
            
            proba = self.model.predict_proba(processed_data)[0][1]
            pred = self.model.predict(processed_data)[0]
            
            return {
                'probability': proba,
                'prediction': pred,
                'risk': 'Élevé' if proba > 0.7 else 'Modéré' if proba > 0.4 else 'Faible'
            }
        except Exception as e:
            st.error(f"Erreur prédiction: {e}")
            return None

def main():
    st.title("🔮 Prédicteur de Churn Client")
    
    # Initialiser le modèle
    predictor = ChurnPredictor()
    
    # Afficher les features attendues (pour debug)
    with st.sidebar.expander("🔧 Debug Info"):
        st.write(f"Features attendues: {len(predictor.feature_names)}")
        st.write("5 premières:", predictor.feature_names[:5])
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Prédiction", "Batch", "Aide"])
    
    if page == "Prédiction":
        show_prediction_page(predictor)
    elif page == "Batch":
        show_batch_page(predictor)
    else:
        show_help_page()

def show_prediction_page(predictor):
    st.header("📋 Informations Client")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Profil")
        gender = st.selectbox("Genre", ["Female", "Male"])
        senior = st.radio("Senior Citizen", [0, 1], format_func=lambda x: "Oui" if x else "Non")
        partner = st.selectbox("Partenaire", ["No", "Yes"])
        dependents = st.selectbox("Dépendants", ["No", "Yes"])
    
    with col2:
        st.subheader("Contrat & Services")
        tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
        contract = st.selectbox("Type de Contrat", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Service Internet", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Méthode de Paiement", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    
    with col3:
        st.subheader("Coûts & Autres")
        monthly = st.number_input("Charges Mensuelles ($)", 10.0, 200.0, 70.0)
        total = st.number_input("Charges Totales ($)", 0.0, 10000.0, 1000.0)
        paperless = st.selectbox("Facture Électronique", ["Yes", "No"])
        phone = st.selectbox("Téléphonie", ["Yes", "No"])
    
    # Valeurs par défaut pour les autres champs
    default_data = {
        'MultipleLines': 'No',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No', 
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No'
    }
    
    # Données complètes du client
    customer_data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': default_data['MultipleLines'],
        'InternetService': internet,
        'OnlineSecurity': default_data['OnlineSecurity'],
        'OnlineBackup': default_data['OnlineBackup'],
        'DeviceProtection': default_data['DeviceProtection'],
        'TechSupport': default_data['TechSupport'],
        'StreamingTV': default_data['StreamingTV'],
        'StreamingMovies': default_data['StreamingMovies'],
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly,
        'TotalCharges': total
    }
    
    # Bouton de prédiction
    if st.button("🎯 Prédire le Risque de Churn", type="primary", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            result = predictor.predict(customer_data)
        
        if result:
            show_prediction_results(result, customer_data)

    st.markdown("""Pour assistance technique, contactez Osséni SOSSOU à osseni.sossou@imsp-uac.org.
    """)
    
def show_prediction_results(result, customer_data):
    st.markdown("---")
    st.header("📊 Résultats de la Prédiction")
    
    proba = result['probability']
    risk = result['risk']
    prediction = result['prediction']
    
    # Métriques en haut
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Probabilité de Churn", f"{proba:.1%}")
    
    with col2:
        risk_color = {"Élevé": "red", "Modéré": "orange", "Faible": "green"}
        st.metric("Niveau de Risque", risk)
    
    with col3:
        status = "⚠️ CLIENT À RISQUE" if prediction == 1 else "✅ CLIENT FIDÈLE"
        st.metric("Recommandation", status)
    
    with col4:
        confidence = "Élevée" if proba > 0.8 or proba < 0.2 else "Moyenne"
        st.metric("Confiance", confidence)
    
    # Barre de progression colorée
    st.subheader("Score de Risque")
    
    # Créer une barre de progression colorée
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.barh([0], [1], color='lightgray', alpha=0.3)
    ax.barh([0], [proba], color='red' if proba > 0.5 else 'green', alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilité de Churn')
    ax.set_yticks([])
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    ax.text(proba, 0, f'{proba:.1%}', ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    st.pyplot(fig)
    
    # Recommandations détaillées
    st.subheader("💡 Plan d'Action Recommandé")
    
    if risk == "Élevé":
        st.error("""
        **🚨 ACTION IMMÉDIATE REQUISE**
        
        **Priorité 1 - Contact Urgent:**
        - 📞 Appel téléphonique sous 24h
        - 👥 Escalade vers responsable clientèle
        - 🎯 Analyse root cause du mécontentement
        
        **Priorité 2 - Rétention:**
        - 💰 Offre commerciale personnalisée
        - 🔄 Révision du contrat actuel  
        - 🎁 Avantage fidélité immédiat
        
        **Priorité 3 - Suivi:**
        - 📊 Surveillance quotidienne
        - 🔔 Alertes proactive
        - 📝 Rapport détaillé
        """)
    
    elif risk == "Modéré":
        st.warning("""
        **⚠️ SURVEILLANCE ACTIVE**
        
        **Actions à 7 jours:**
        - 📧 Email de vérification satisfaction
        - 📞 Rappel de service client
        - 🔄 Proposition de services additionnels
        
        **Actions préventives:**
        - 📊 Monitoring bi-hebdomadaire
        - 🎯 Offres ciblées
        - 📋 Revue de compte
        
        **Mesures de fidélisation:**
        - ⭐ Programme de parrainage
        - 🏆 Avantages membre
        - 🔍 Feedback continu
        """)
    
    else:
        st.success("""
        **✅ FIDÉLISATION & CROISSANCE**
        
        **Renforcement relation:**
        - 📱 Communication régulière
        - 🎁 Offres personnalisées
        - ⭐ Programme premium
        
        **Développement:**
        - 🔄 Upselling services
        - 👥 Programme de recommandation
        - 📈 Analyse besoins futurs
        
        **Rétention proactive:**
        - 📊 Revue trimestrielle
        - 🎯 Enquêtes satisfaction
        - 💡 Innovations partagées
        """)
    
    # Analyse détaillée
    with st.expander("🔍 Analyse Technique Détaillée"):
        show_technical_analysis(customer_data, proba)


def show_technical_analysis(customer_data, probability):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Facteurs de Risque")
        factors = []
        
        # Analyse des facteurs de risque
        if customer_data['Contract'] == 'Month-to-month':
            factors.append(("Contrat mensuel", "+++ Risque élevé"))
        elif customer_data['Contract'] == 'One year':
            factors.append(("Contrat 1 an", "+ Risque modéré"))
        else:
            factors.append(("Contrat 2 ans", "- Risque faible"))
        
        if customer_data['tenure'] < 6:
            factors.append(("Ancienneté < 6 mois", "+++ Risque très élevé"))
        elif customer_data['tenure'] < 12:
            factors.append(("Ancienneté < 1 an", "++ Risque élevé"))
        elif customer_data['tenure'] < 24:
            factors.append(("Ancienneté 1-2 ans", "+ Risque modéré"))
        else:
            factors.append(("Ancienneté > 2 ans", "- Risque faible"))
        
        if customer_data['InternetService'] == 'Fiber optic':
            factors.append(("Fibre optique", "++ Risque élevé"))
        elif customer_data['InternetService'] == 'DSL':
            factors.append(("DSL", "+ Risque modéré"))
        else:
            factors.append(("Pas d'internet", "- Risque faible"))
        
        if customer_data['PaymentMethod'] == 'Electronic check':
            factors.append(("Paiement électronique", "++ Risque élevé"))
        
        if customer_data['OnlineSecurity'] == 'No' and customer_data['InternetService'] != 'No':
            factors.append(("Pas de sécurité", "++ Risque élevé"))
        
        for factor, impact in factors:
            st.write(f"• **{factor}**: {impact}")
    
    with col2:
        st.subheader("Indicateurs Clés")
        
        # Calculer les ratios
        charge_ratio = customer_data['MonthlyCharges'] / (customer_data['tenure'] + 1)
        total_ratio = customer_data['TotalCharges'] / (customer_data['MonthlyCharges'] + 1)
        
        metrics = {
            "Ancienneté": f"{customer_data['tenure']} mois",
            "Ratio Charges/Ancienneté": f"{charge_ratio:.2f}",
            "Ratio Total/Mensuel": f"{total_ratio:.2f}",
            "Type de contrat": customer_data['Contract'],
            "Service principal": customer_data['InternetService'],
            "Support technique": customer_data['TechSupport']
        }
        
        for key, value in metrics.items():
            st.write(f"**{key}:** {value}")
        
        # Graphique de probabilité
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(['Probabilité'], [probability], color='red' if probability > 0.5 else 'green')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Score')
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
        ax.text(probability, 0, f'{probability:.1%}', ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
        st.pyplot(fig)

def show_batch_page(predictor):
    st.header("📊 Analyse par Lot")
    
    st.info("""
    **Format requis:** CSV avec colonnes:
    - gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, InternetService
    - Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
    """)
    
    uploaded_file = st.file_uploader("Télécharger CSV", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ {len(df)} clients chargés")
            
            if st.button("🎯 Analyser le Lot", type="primary"):
                results = []
                with st.spinner("Analyse en cours..."):
                    for idx, row in df.iterrows():
                        result = predictor.predict(row.to_dict())
                        if result:
                            results.append({
                                'Client': idx + 1,
                                'Probabilité': result['probability'],
                                'Risque': result['risk'],
                                'Prédiction': 'Churn' if result['prediction'] else 'Fidèle'
                            })
                
                if results:
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    
                    # Statistiques
                    st.subheader("📈 Statistiques du Lot")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Taux de Churn", f"{(results_df['Prédiction'] == 'Churn').mean():.1%}")
                    with col2:
                        st.metric("Risque Moyen", f"{results_df['Probabilité'].mean():.1%}")
                    with col3:
                        high_risk = (results_df['Risque'] == 'Élevé').sum()
                        st.metric("Risques Élevés", high_risk)
        
        except Exception as e:
            st.error(f"Erreur: {e}")
    st.markdown("""Pour assistance technique, contactez Osséni SOSSOU à osseni.sossou@imsp-uac.org.
    """)
def show_help_page():
    st.header("📚 Guide d'Utilisation")
    
    st.markdown("""
    ## 🎯 Comment Utiliser
    
    **Prédiction Unique:**
    1. Remplir toutes les informations client
    2. Cliquer sur "Prédire le Risque de Churn"  
    3. Consulter les résultats et recommandations
    
    **Analyse par Lot:**
    1. Préparer un fichier CSV formaté
    2. Télécharger le fichier
    3. Lancer l'analyse batch
    
    ## 📊 Échelle de Risque
    
    | Probabilité | Niveau | Action |
    |------------|---------|---------|
    | < 40% | 🟢 Faible | Fidélisation |
    | 40-70% | 🟡 Modéré | Surveillance |
    | > 70% | 🔴 Élevé | Intervention |
    
    ## 🔍 Facteurs Clés
    
    Le modèle analyse:
    - **Ancienneté** et historique
    - **Type de contrat** (mensuel/annuel)
    - **Services souscrits**
    - **Méthodes de paiement**
    - **Ratios financiers**
    
    ## 🛠️ Support
    """)

    st.markdown("""Pour assistance technique, contactez Osséni SOSSOU à osseni.sossou@imsp-uac.org.
    """)
if __name__ == "__main__":
    main()
