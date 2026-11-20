"""Heart Disease KNN Explorer — Streamlit Community Cloud ready."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, classification_report)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="HeartLab | KNN Classification", page_icon="🫀", layout="wide", initial_sidebar_state="expanded")

FEATURES = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]
TARGET = "HeartDisease"
CATEGORICAL = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUMERIC = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
LABELS = {0:"ไม่พบโรคหัวใจ", 1:"พบโรคหัวใจ"}
DESCRIPTIONS = {
    "Age":"อายุ (ปี)", "Sex":"เพศ (ค่ารหัส)", "ChestPainType":"ประเภทอาการเจ็บหน้าอก (ค่ารหัส)",
    "RestingBP":"ความดันขณะพัก (mmHg)", "Cholesterol":"คอเลสเตอรอล (mg/dL)",
    "FastingBS":"น้ำตาลขณะอดอาหาร (ค่ารหัส)", "RestingECG":"ผลคลื่นไฟฟ้าหัวใจ (ค่ารหัส)",
    "MaxHR":"อัตราการเต้นหัวใจสูงสุด (ครั้ง/นาที)", "ExerciseAngina":"อาการเจ็บหน้าอกเมื่อออกกำลังกาย (ค่ารหัส)",
    "Oldpeak":"ค่า ST depression (Oldpeak)", "ST_Slope":"ความชัน ST segment (ค่ารหัส)",
}

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background:linear-gradient(135deg,#f5f8fd,#eef4fc 52%,#f9fbff);}
[data-testid="stSidebar"] {background:#0d1730;}
[data-testid="stSidebar"] * {color:#eaf1ff !important;}
[data-testid="stSidebar"] .stButton button {background:#2058d4; color:white;}
h1,h2,h3 {color:#11264a;letter-spacing:-.025em;}
.hero {background:linear-gradient(115deg,#122755 0%,#264b9e 65%,#4f75cd 100%); color:#fff; border-radius:22px; padding:32px 35px; margin-bottom:20px; box-shadow:0 12px 30px rgba(29,64,128,.16)}
.hero h1 {color:#fff;font-size:2.2rem;margin:0 0 8px 0;}
.hero p {color:#dce9ff;margin:0;font-size:1rem;}
.pill {display:inline-block;border:1px solid #92ace3;background:#294c8f;color:#fff;padding:5px 12px;border-radius:24px;font-size:.75rem;margin-bottom:12px}
[data-testid="stMetric"] {background:#fff;border:1px solid #e1e9f6;border-radius:15px;padding:15px;box-shadow:0 4px 14px rgba(37,65,106,.04)}
[data-testid="stMetricLabel"] {color:#587092;}
[data-testid="stMetricValue"] {color:#162f61;}
div[data-testid="stAlert"]{border-radius:12px;}
section[data-testid="stSidebar"] hr {border-color:#37476a}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_data(data:bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(data), encoding="utf-8-sig")

def check_data(frame:pd.DataFrame, with_target=True):
    needed = FEATURES + ([TARGET] if with_target else [])
    missing = [col for col in needed if col not in frame.columns]
    if missing:
        raise ValueError("ไม่พบคอลัมน์ที่จำเป็น: " + ", ".join(missing))
    df = frame[needed].copy()
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL:
        invalid = df[col].notna() & (~df[col].isin({"Sex":[0,1], "ChestPainType":[1,2,3,4], "FastingBS":[0,1], "RestingECG":[1,2,3], "ExerciseAngina":[0,1], "ST_Slope":[1,2,3]}[col]))
        if invalid.any():
            raise ValueError(f"คอลัมน์ {col} มีค่ารหัสนอกช่วงที่รองรับ ({int(invalid.sum())} แถว)")
    if with_target:
        if df[TARGET].isna().any() or not df[TARGET].isin([0,1]).all():
            raise ValueError("HeartDisease ต้องเป็น 0 หรือ 1 และต้องไม่มีค่าว่าง")
        df[TARGET] = df[TARGET].astype(int)
        if df[TARGET].nunique()<2:
            raise ValueError("ต้องมีตัวอย่างทั้งคลาส 0 และ 1 สำหรับการฝึกโมเดล")
    return df

def make_pipeline(k=5, weights='uniform'):
    numeric_pipe = Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
    categorical_pipe = Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                                 ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    prep = ColumnTransformer([('numerical',numeric_pipe,NUMERIC),('categorical',categorical_pipe,CATEGORICAL)])
    return Pipeline([('preprocess',prep),('knn',KNeighborsClassifier(n_neighbors=k,weights=weights))])

@st.cache_resource(show_spinner="กำลังฝึกโมเดลและประเมินผล...")
def train_model(data:bytes, test_fraction:float, seed:int, max_k:int, weights:str, folds:int, auto_tune:bool):
    df = check_data(read_data(data))
    X = df[FEATURES].copy()
    # Only values considered physiologically invalid are treated as missing.
    X[['RestingBP','Cholesterol']] = X[['RestingBP','Cholesterol']].replace(0,np.nan)
    y = df[TARGET]
    if y.value_counts().min()<folds+1:
        raise ValueError("จำนวนข้อมูลแต่ละคลาสน้อยเกินไปสำหรับ Cross-validation ที่เลือก")
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=test_fraction,stratify=y,random_state=seed)
    smallest_class = int(y_train.value_counts().min())
    if smallest_class < folds:
        raise ValueError("จำนวนตัวอย่างในชุดฝึกไม่พอสำหรับจำนวน folds ที่เลือก")
    min_cv_train_size = len(x_train) - int(np.ceil(len(x_train)/folds))
    candidate_k = list(range(1,min(max_k,min_cv_train_size)+1,2))
    if not candidate_k:
        raise ValueError("ชุดข้อมูลมีขนาดเล็กเกินไป")
    cv = StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    if auto_tune:
        grid = GridSearchCV(make_pipeline(weights=weights),{'knn__n_neighbors':candidate_k},cv=cv,
                            scoring='balanced_accuracy',n_jobs=-1,refit=True,error_score='raise')
        grid.fit(x_train,y_train)
        model=grid.best_estimator_
        best_k=int(grid.best_params_['knn__n_neighbors'])
        cv_results=pd.DataFrame(grid.cv_results_)[['param_knn__n_neighbors','mean_test_score','std_test_score']]
        cv_results.columns=['k','mean_score','std_score']
        cv_results['k']=cv_results['k'].astype(int)
    else:
        best_k=min(max_k,min_cv_train_size)
        model=make_pipeline(k=best_k,weights=weights).fit(x_train,y_train)
        cv_results=pd.DataFrame()
    prediction=model.predict(x_test)
    prob=model.predict_proba(x_test)[:,list(model.classes_).index(1)]
    scores={
        'accuracy': accuracy_score(y_test,prediction),
        'balanced_accuracy':balanced_accuracy_score(y_test,prediction),
        'precision':precision_score(y_test,prediction,zero_division=0),
        'recall':recall_score(y_test,prediction,zero_division=0),
        'f1':f1_score(y_test,prediction,zero_division=0),
        'auc':roc_auc_score(y_test,prob) if y_test.nunique()==2 else float('nan')}
    outcome=x_test.copy()
    outcome[TARGET]=y_test.to_numpy()
    outcome['Predicted']=prediction
    outcome['Probability_HeartDisease_1']=prob
    return model, scores, best_k, cv_results, y_test.to_numpy(), prediction, prob, outcome, len(x_train),len(x_test)

with st.sidebar:
    st.markdown("## 🫀 HeartLab")
    st.caption("KNN Classification Studio")
    st.divider()
    st.markdown("#### 📁 ชุดข้อมูล")
    source=st.radio("เลือกแหล่งข้อมูล",["ใช้ Heart3.csv ที่มาพร้อมแอป","อัปโหลดไฟล์ CSV"],label_visibility='collapsed')
    default_file=Path(__file__).parent / '/data/Heart3.csv'
    uploaded=None
    if source=="อัปโหลดไฟล์ CSV":
        uploaded=st.file_uploader("อัปโหลดไฟล์ข้อมูล",type=['csv'])
        if uploaded is None:
            st.info("อัปโหลด CSV เพื่อใช้งาน หรือเลือกชุดข้อมูลเริ่มต้น")
    st.divider()
    st.markdown("#### ⚙️ ตั้งค่าโมเดล")
    test_fraction=st.slider("สัดส่วนชุดทดสอบ",10,40,20,5)/100
    seed=st.number_input("Random state",value=42,min_value=0,max_value=99999)
    auto_tune=st.toggle("เลือกค่า K ด้วย Cross-validation",value=True)
    max_k=st.slider("ค่า K สูงสุดที่พิจารณา" if auto_tune else "จำนวนเพื่อนบ้าน (K)",1,41,21,2)
    folds=st.select_slider("จำนวน Cross-validation folds",options=[3,5,10],value=5,disabled=not auto_tune)
    weights=st.selectbox("น้ำหนักเพื่อนบ้าน",['uniform','distance'],format_func=lambda s:'Uniform (เท่ากัน)' if s=='uniform' else 'Distance (ใกล้มีน้ำหนักมาก)')
    st.divider()
    st.caption("เพื่อการศึกษาและการทดลอง ไม่ใช่ระบบวินิจฉัยทางการแพทย์")

st.markdown('''<div class="hero"><span class="pill">MACHINE LEARNING • INTERACTIVE DASHBOARD</span><h1>Heart Disease Classification</h1><p>สำรวจข้อมูล • สร้างโมเดล KNN • ประเมินผล • ทำนายข้อมูลใหม่ ด้วย Streamlit</p></div>''',unsafe_allow_html=True)

try:
    raw_bytes=uploaded.getvalue() if uploaded is not None else default_file.read_bytes()
    df=check_data(read_data(raw_bytes))
    if df[TARGET].value_counts().min()<3:
        st.error("แต่ละคลาสควรมีข้อมูลอย่างน้อย 3 รายการสำหรับการแบ่งข้อมูลและประเมินโมเดล")
        st.stop()
except Exception as exc:
    st.error(f"อ่านข้อมูลไม่สำเร็จ: {exc}")
    st.stop()

try:
    model, scores, best_k, cv_results, y_test, prediction, prob, outcome, n_train,n_test = train_model(raw_bytes,test_fraction,int(seed),max_k,weights,int(folds),auto_tune)
except Exception as exc:
    st.error(f"ไม่สามารถฝึกโมเดลได้: {exc}")
    st.stop()

t1,t2,t3,t4=st.tabs(["📊 ภาพรวมข้อมูล","🧠 ฝึกและประเมิน KNN","🔍 ทำนายรายบุคคล","📂 ทำนายหลายรายการ"])

with t1:
    st.subheader('ภาพรวมชุดข้อมูล')
    a,b,c,d=st.columns(4)
    a.metric('จำนวนข้อมูล',f'{len(df):,}')
    b.metric('ตัวแปรทำนาย',len(FEATURES))
    c.metric('พบโรคหัวใจ',f'{(df[TARGET]==1).sum():,}',f'{(df[TARGET]==1).mean():.1%}')
    d.metric('ค่าว่างในไฟล์',f'{int(df.isna().sum().sum()):,}')
    left,right=st.columns(2)
    with left:
        counts=df[TARGET].value_counts().sort_index().rename(index=LABELS).reset_index()
        counts.columns=['กลุ่ม','จำนวน']
        fig=px.pie(counts,names='กลุ่ม',values='จำนวน',hole=.58,color='กลุ่ม',color_discrete_sequence=['#26a69a','#ed6877'],title='สัดส่วนคลาสเป้าหมาย')
        fig.update_layout(margin=dict(l=10,r=10,t=60,b=10),legend_orientation='h',legend_y=-.12)
        st.plotly_chart(fig,use_container_width=True)
    with right:
        fig=px.histogram(df,x='Age',color=df[TARGET].map(LABELS),barmode='overlay',nbins=20,
                         color_discrete_sequence=['#2677c9','#eb7081'],title='การกระจายอายุจำแนกตามกลุ่ม',labels={'color':'กลุ่ม','Age':'อายุ'})
        fig.update_layout(margin=dict(l=10,r=10,t=60,b=10),legend_title_text='กลุ่ม')
        st.plotly_chart(fig,use_container_width=True)
    x,y=st.columns(2)
    with x:
        feature=st.selectbox('เลือกตัวแปรเพื่อสำรวจ',FEATURES,index=4,format_func=lambda f:f'{f} — {DESCRIPTIONS[f]}')
        if feature in CATEGORICAL:
            count=df.groupby([feature,TARGET]).size().reset_index(name='จำนวน')
            count['กลุ่ม']=count[TARGET].map(LABELS)
            fig=px.bar(count,x=feature,y='จำนวน',color='กลุ่ม',barmode='group',title=f'การกระจาย {feature}',color_discrete_sequence=['#2b80cf','#ef7280'])
        else:
            temp=df.copy();temp['กลุ่ม']=temp[TARGET].map(LABELS)
            fig=px.histogram(temp,x=feature,color='กลุ่ม',barmode='overlay',nbins=24,title=f'การกระจาย {feature}',color_discrete_sequence=['#2b80cf','#ef7280'])
        st.plotly_chart(fig,use_container_width=True)
    with y:
        st.markdown('#### การตรวจสอบคุณภาพข้อมูล')
        st.write(f"ข้อมูลซ้ำทั้งแถว: **{df.duplicated().sum():,}** แถว")
        st.write(f"RestingBP เท่ากับ 0: **{(df['RestingBP']==0).sum():,}** แถว")
        st.write(f"Cholesterol เท่ากับ 0: **{(df['Cholesterol']==0).sum():,}** แถว")
        st.info('โมเดลถือค่า 0 ของ RestingBP และ Cholesterol เป็นค่าสูญหาย แล้วแทนด้วยค่ามัธยฐานที่เรียนรู้จากชุดฝึกเท่านั้น')
        st.info('ตัวแปรที่เป็นรหัสหมวดหมู่ใช้ One-Hot Encoding เพื่อไม่บังคับให้ระยะห่างของรหัส 1, 2, 3 มีความหมายเชิงปริมาณ')
    with st.expander('ดูรายละเอียดฟิลด์และตัวอย่างข้อมูล'):
        st.dataframe(pd.DataFrame({'คอลัมน์':FEATURES+[TARGET], 'คำอธิบาย':[DESCRIPTIONS[k] for k in FEATURES]+['คลาสเป้าหมาย 0/1']}),hide_index=True,use_container_width=True)
        st.dataframe(df.head(30),hide_index=True,use_container_width=True)
        st.download_button('ดาวน์โหลดไฟล์ที่ใช้งาน',df.to_csv(index=False).encode('utf-8-sig'),'heart_data.csv','text/csv')

with t2:
    st.subheader('ผลการฝึกและประเมินโมเดล')
    st.caption(f'Stratified hold-out: Train {n_train:,} / Test {n_test:,} • Selected K = {best_k} • Weights = {weights}')
    m=st.columns(6)
    for box,(name,val) in zip(m,[('Accuracy',scores['accuracy']),('Balanced Acc.',scores['balanced_accuracy']),('Precision',scores['precision']),('Recall',scores['recall']),('F1-score',scores['f1']),('ROC-AUC',scores['auc'])]):
        box.metric(name,f'{val:.3f}')
    st.info('การปรับค่า K ด้วย Cross-validation ใช้เฉพาะชุดฝึก และประเมินค่าด้านบนด้วยชุดทดสอบที่แยกไว้ ไม่ได้นำชุดทดสอบไปเลือก K')
    left,right=st.columns(2)
    with left:
        cm=confusion_matrix(y_test,prediction,labels=[0,1])
        fig=go.Figure(go.Heatmap(z=cm,x=['ทำนาย 0','ทำนาย 1'],y=['จริง 0','จริง 1'],colorscale=[[0,'#ecf3fc'],[1,'#2a5ab3']],showscale=False,text=cm,texttemplate='%{text}',textfont={'size':23}))
        fig.update_layout(title='Confusion Matrix',xaxis_title='Predicted',yaxis_title='Actual',height=370,margin=dict(l=20,r=20,t=60,b=20))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        fpr,tpr,_=roc_curve(y_test,prob)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',name=f'KNN (AUC {scores["auc"]:.3f})',line=dict(color='#245fd0',width=4)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Random',line=dict(color='#adb8c8',dash='dash')))
        fig.update_layout(title='ROC Curve',xaxis_title='False Positive Rate',yaxis_title='True Positive Rate',height=370,margin=dict(l=20,r=20,t=60,b=20))
        st.plotly_chart(fig,use_container_width=True)
    if auto_tune and not cv_results.empty:
        fig=px.line(cv_results,x='k',y='mean_score',markers=True,title='ประสิทธิภาพเฉลี่ยจาก Cross-validation ตามค่า K',labels={'k':'จำนวนเพื่อนบ้าน (K)','mean_score':'Mean Balanced Accuracy'})
        fig.add_vline(x=best_k,line_dash='dash',line_color='#e26066')
        fig.update_traces(line_color='#235fcc')
        st.plotly_chart(fig,use_container_width=True)
        with st.expander('ดูตารางผล Cross-validation'):
            st.dataframe(cv_results,hide_index=True,use_container_width=True)
    with st.expander('Classification Report'):
        report=classification_report(y_test,prediction,labels=[0,1],target_names=['No heart disease (0)','Heart disease (1)'],output_dict=True,zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose(),use_container_width=True)
    st.download_button('⬇️ ดาวน์โหลดผลการทำนายชุดทดสอบ',outcome.to_csv(index=False).encode('utf-8-sig'),'knn_test_predictions.csv','text/csv')

with t3:
    st.subheader('ทำนายข้อมูลผู้รับการประเมิน 1 รายการ')
    st.caption('เลือกรหัสหมวดหมู่ให้สอดคล้องกับข้อมูลต้นฉบับ (ไฟล์ไม่ได้แนบตารางคำอธิบาย mapping ของรหัส)')
    med=df[FEATURES].median(numeric_only=True)
    with st.form('patient_form'):
        ca,cb,cc=st.columns(3)
        with ca:
            age=st.number_input('Age — อายุ (ปี)',min_value=1,max_value=120,value=int(med['Age']))
            sex=st.selectbox('Sex — รหัสเพศ',[0,1],index=int(med['Sex']))
            chest=st.selectbox('ChestPainType — รหัสอาการเจ็บหน้าอก',[1,2,3,4])
            bp=st.number_input('RestingBP (mmHg)',min_value=0,max_value=300,value=int(med['RestingBP']))
        with cb:
            chol=st.number_input('Cholesterol (mg/dL)',min_value=0,max_value=800,value=int(med['Cholesterol']))
            fbs=st.selectbox('FastingBS — รหัสน้ำตาลขณะอดอาหาร',[0,1])
            ecg=st.selectbox('RestingECG — รหัสผล ECG',[1,2,3],index=1)
            maxhr=st.number_input('MaxHR (ครั้ง/นาที)',min_value=30,max_value=250,value=int(med['MaxHR']))
        with cc:
            angina=st.selectbox('ExerciseAngina — รหัสอาการเจ็บหน้าอกเวลาออกกำลังกาย',[0,1])
            oldpeak=st.number_input('Oldpeak',min_value=-5.0,max_value=10.0,value=float(med['Oldpeak']),step=0.1,format='%.1f')
            slope=st.selectbox('ST_Slope — รหัสความชัน ST',[1,2,3],index=1)
        do_predict=st.form_submit_button('🔍 ประมวลผลและทำนาย',type='primary',use_container_width=True)
    if do_predict:
        sample=pd.DataFrame([[age,sex,chest,bp,chol,fbs,ecg,maxhr,angina,oldpeak,slope]],columns=FEATURES)
        sample[['RestingBP','Cholesterol']]=sample[['RestingBP','Cholesterol']].replace(0,np.nan)
        proba=float(model.predict_proba(sample)[0,list(model.classes_).index(1)])
        cls=int(model.predict(sample)[0]);label=LABELS[cls]
        st.markdown('### ผลการจำแนก')
        a,b=st.columns(2)
        a.metric('คลาสที่โมเดลทำนาย',f'{cls} — {label}')
        b.metric('Model score สำหรับคลาส 1',f'{proba:.1%}')
        st.progress(float(proba),text='สัดส่วนคะแนนการโหวตแบบปรับน้ำหนักของเพื่อนบ้านสำหรับคลาส 1')
        st.warning('ผลนี้เป็นเพียงการทำนายเชิงสถิติจากข้อมูลฝึก ไม่ใช่การวินิจฉัยทางการแพทย์ และค่าคะแนนอาจไม่ใช่ความน่าจะเป็นของโรคที่ปรับเทียบแล้ว')

with t4:
    st.subheader('ทำนายข้อมูลครั้งละหลายรายการ (Batch Prediction)')
    st.write('ไฟล์สำหรับทำนายต้องมีชื่อคอลัมน์ตัวแปรทั้ง 11 ตัวเหมือนข้อมูลฝึก โดยไม่จำเป็นต้องมีคอลัมน์ HeartDisease')
    template=df[FEATURES].head(5)
    st.download_button('⬇️ ดาวน์โหลด CSV ตัวอย่างสำหรับทำนาย',template.to_csv(index=False).encode('utf-8-sig'),'knn_prediction_template.csv','text/csv')
    batchfile=st.file_uploader('อัปโหลด CSV สำหรับทำนาย',type=['csv'],key='batch')
    if batchfile is not None:
        try:
            batchraw=read_data(batchfile.getvalue())
            sample=check_data(batchraw,with_target=False)
            if sample.empty:
                raise ValueError('ไฟล์ไม่มีรายการข้อมูล')
            to_predict=sample.copy()
            to_predict[['RestingBP','Cholesterol']]=to_predict[['RestingBP','Cholesterol']].replace(0,np.nan)
            pred=model.predict(to_predict)
            confidence=model.predict_proba(to_predict)[:,list(model.classes_).index(1)]
            results=batchraw.copy()
            results['Predicted_HeartDisease']=pred
            results['Probability_HeartDisease_1']=np.round(confidence,5)
            results['Predicted_Label']=[LABELS[int(p)] for p in pred]
            c1,c2,c3=st.columns(3)
            c1.metric('ทั้งหมด',len(results));c2.metric('ทำนายคลาส 0',int((pred==0).sum()));c3.metric('ทำนายคลาส 1',int((pred==1).sum()))
            st.dataframe(results.head(100),use_container_width=True,hide_index=True)
            st.download_button('⬇️ ดาวน์โหลดผลทำนายทั้งหมด',results.to_csv(index=False).encode('utf-8-sig'),'knn_batch_predictions.csv','text/csv',type='primary')
        except Exception as exc:
            st.error(f'ไม่สามารถทำนายไฟล์นี้ได้: {exc}')

st.divider()
st.caption('HeartLab • KNN + StandardScaler + OneHotEncoder • Data leakage–safe preprocessing • For educational purposes only')
