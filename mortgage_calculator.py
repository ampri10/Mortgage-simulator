import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

def calculate_mortgage(years, debt, r, inflation_linked,
                      years2, debt2, r2, inflation_linked2,
                      years3, debt3, r3, inflation_linked3,
                      inflation_rate):
    """
    Calculate mortgage payments for up to 3 loan buckets
    """
    Loan = debt
    Loan2 = debt2
    Loan3 = debt3
    
    total_debts = debt + debt2 + debt3
    
    df = pd.DataFrame(columns=['year', 'pmt_m', 'pmt', 'keren', 'rate', 'debt'])
    df.loc[len(df.index)] = [0, None, None, 0, 0, total_debts]
    
    tot_years = max(years, years2, years3)
    
    for n in range(1, tot_years + 1):
        # Apply inflation to rates if linked
        current_r = r + (inflation_rate if inflation_linked else 0)
        current_r2 = r2 + (inflation_rate if inflation_linked2 else 0)
        current_r3 = r3 + (inflation_rate if inflation_linked3 else 0)
        
        # Monthly payments
        if n > years:
            pmt_m1 = 0
        else:
            pmt_m1 = -npf.pmt(current_r/12, years*12, Loan, 0, 0)
            
        if n > years2:
            pmt_m2 = 0
        else:
            pmt_m2 = -npf.pmt(current_r2/12, years2*12, Loan2, 0, 0)
            
        if n > years3:
            pmt_m3 = 0
        else:
            pmt_m3 = -npf.pmt(current_r3/12, years3*12, Loan3, 0, 0)
            
        pmt_m = round(pmt_m1 + pmt_m2 + pmt_m3, 2)
        
        # Annual payments
        if n > years:
            pmt1 = 0
        else:
            pmt1 = -npf.pmt(current_r, years, Loan, 0, 0)
            
        if n > years2:
            pmt2 = 0
        else:
            pmt2 = -npf.pmt(current_r2, years2, Loan2, 0, 0)
            
        if n > years3:
            pmt3 = 0
        else:
            pmt3 = -npf.pmt(current_r3, years3, Loan3, 0, 0)
            
        pmt = round(pmt1 + pmt2 + pmt3, 2)
        
        # Principal payments
        if n > years:
            keren1 = 0
        else:
            keren1 = -npf.ppmt(current_r, n, years, Loan, 0, 0)
            
        if n > years2:
            keren2 = 0
        else:
            keren2 = -npf.ppmt(current_r2, n, years2, Loan2, 0, 0)
            
        if n > years3:
            keren3 = 0
        else:
            keren3 = -npf.ppmt(current_r3, n, years3, Loan3, 0, 0)
            
        keren = round(keren1 + keren2 + keren3, 2)
        
        # Interest payments
        if n > years:
            rate1 = 0
        else:
            rate1 = -npf.ipmt(current_r, n, years, Loan, 0, 0).item()
            
        if n > years2:
            rate2 = 0
        else:
            rate2 = -npf.ipmt(current_r2, n, years2, Loan2, 0, 0).item()
            
        if n > years3:
            rate3 = 0
        else:
            rate3 = -npf.ipmt(current_r3, n, years3, Loan3, 0, 0).item()
            
        rate = round(rate1 + rate2 + rate3, 2)
        
        total_debts = round(total_debts - keren, 2)
        
        df.loc[len(df.index)] = [n, pmt_m, pmt, keren, rate, total_debts]
    
    return df

def create_charts(df):
    """Create all four charts for the mortgage analysis"""
    
    # 1. Data Table
    table_fig = ff.create_table(df, height_constant=60)
    table_fig.update_layout(
        autosize=False,
        width=600,
        height=400,
        title="Mortgage Payment Schedule"
    )
    
    # 2. Pie Chart - Principal vs Interest
    colors = ['gold', 'darkorange']
    pie_fig = go.Figure(data=[go.Pie(
        labels=['Principal (Keren)', 'Interest (Rate)'],
        values=[sum(df['keren']), sum(df['rate'])]
    )])
    pie_fig.update_layout(
        autosize=False,
        width=600,
        height=400,
        title="Total Principal vs Interest"
    )
    pie_fig.update_traces(
        hoverinfo='label+percent',
        textinfo='value',
        textfont_size=16,
        marker=dict(colors=colors, line=dict(color='#000000', width=2))
    )
    
    # 3. Line Chart - Debt Progress and Monthly Payments
    subfig = make_subplots(specs=[[{"secondary_y": True}]])

    #df["debt_label"] = df["debt"].apply(lambda x: f"{x/1000:,.1f} k".replace(".", ",").replace(",", "X", 1).replace(".", ",").replace("X", "."))
    debt_labels = [
    f"{x/1000:,.1f} k".replace(".", ",").replace(",", "X", 1).replace(".", ",").replace("X", ".") if i % 2 == 0 else ""
    for i, x in enumerate(df["debt"])
]
    
    # Debt line
    fig1 = px.line(df, x="year", y="debt", text=debt_labels, markers=True)
    fig1.update_traces(textposition="top right", name="Total Debt")
    
    # Monthly payment line
    fig2 = px.line(df, x="year", y="pmt_m", markers=True)
    fig2.update_traces(yaxis="y2", name="Monthly Payment")
    
    subfig.add_traces(fig1.data + fig2.data)
    subfig.layout.title = "Mortgage Progress Over Time"
    subfig.layout.xaxis.title = "Years"
    subfig.layout.yaxis.title = "Total Debt"
    subfig.layout.yaxis2.title = "Monthly Payment"
    subfig.layout.width = 600
    subfig.layout.height = 400
    
    subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
    subfig.update_yaxes(range=[0, max(df['pmt_m']) * 1.2], secondary_y=True)
    
    # 4. Stacked Bar Chart - Annual Principal and Interest
    keren_df = df[['year', 'keren']].copy()
    keren_df['type'] = 'Principal'
    keren_df = keren_df.rename(columns={"keren": "amount"})
    
    rate_df = df[['year', 'rate']].copy()
    rate_df['type'] = 'Interest'
    rate_df = rate_df.rename(columns={"rate": "amount"})
    
    combined_df = pd.concat([keren_df, rate_df])
    
    bar_fig = px.bar(
        combined_df,
        x="year",
        y="amount",
        color='type',
        barmode='group',
        height=400,
        width=600,
        title="Annual Principal vs Interest Payments"
    )
    
    return table_fig, pie_fig, subfig, bar_fig

def main():
    st.set_page_config(page_title="Mortgage Simulator", layout="wide")
    
    st.title("🏠 Advanced Mortgage Simulator")
    st.markdown("---")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Loan Configuration")
        
        # Global inflation rate
        inflation_rate = st.slider(
            "Annual Inflation Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            format="%.1f%%"
        ) / 100
        
        st.markdown("### Loan Bucket 1")
        years1 = st.number_input("Years", min_value=1, max_value=50, value=30, key="years1")
        debt1 = st.number_input("Loan Amount", min_value=0, value=300000, step=10000, key="debt1")
        rate1 = st.slider("Interest Rate (%)", 0.0, 15.0, 4.5, 0.1, key="rate1") / 100
        inflation_linked1 = st.checkbox("Link to Inflation", value=False, key="inf1")
        
        st.markdown("### Loan Bucket 2")
        years2 = st.number_input("Years", min_value=1, max_value=50, value=25, key="years2")
        debt2 = st.number_input("Loan Amount", min_value=0, value=200000, step=10000, key="debt2")
        rate2 = st.slider("Interest Rate (%)", 0.0, 15.0, 3.8, 0.1, key="rate2") / 100
        inflation_linked2 = st.checkbox("Link to Inflation", value=True, key="inf2")
        
        st.markdown("### Loan Bucket 3")
        years3 = st.number_input("Years", min_value=1, max_value=50, value=20, key="years3")
        debt3 = st.number_input("Loan Amount", min_value=0, value=100000, step=10000, key="debt3")
        rate3 = st.slider("Interest Rate (%)", 0.0, 15.0, 5.2, 0.1, key="rate3") / 100
        inflation_linked3 = st.checkbox("Link to Inflation", value=False, key="inf3")
        
        st.markdown("---")
        calculate_btn = st.button("Calculate Mortgage", type="primary")
    
    # Main content area
    if calculate_btn:
        # Calculate mortgage
        df = calculate_mortgage(
            years1, debt1, rate1, inflation_linked1,
            years2, debt2, rate2, inflation_linked2,
            years3, debt3, rate3, inflation_linked3,
            inflation_rate
        )
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Loan Amount",
                f"${debt1 + debt2 + debt3:,.0f}"
            )
        
        with col2:
            st.metric(
                "Total Interest Paid",
                f"${sum(df['rate']):,.0f}"
            )
        
        with col3:
            st.metric(
                "Total Amount Paid",
                f"${sum(df['keren']) + sum(df['rate']):,.0f}"
            )
        
        with col4:
            max_monthly = max(df['pmt_m'])
            st.metric(
                "Peak Monthly Payment",
                f"${max_monthly:,.0f}"
            )
        
        st.markdown("---")
        
        # Create charts
        table_fig, pie_fig, line_fig, bar_fig = create_charts(df)
        
        # Display charts in 2x2 matrix
        st.subheader("📊 Mortgage Analysis Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(table_fig, use_container_width=True)
            st.plotly_chart(pie_fig, use_container_width=True)
        
        with col2:
            st.plotly_chart(line_fig, use_container_width=True)
            st.plotly_chart(bar_fig, use_container_width=True)
        
        # Download option for data
        st.markdown("---")
        st.subheader("📥 Export Data")
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Payment Schedule as CSV",
            data=csv,
            file_name="mortgage_schedule.csv",
            mime="text/csv"
        )
    
    else:
        st.info("👈 Configure your loan parameters in the sidebar and click 'Calculate Mortgage' to see the analysis.")
        
        # Show sample data explanation
        st.markdown("""
        ### How to Use This Mortgage Simulator
        
        1. **Configure up to 3 loan buckets** with different terms and rates
        2. **Set inflation rate** and choose which loans are inflation-linked
        3. **View comprehensive analysis** including:
           - Payment schedule table
           - Principal vs Interest breakdown
           - Debt progress over time
           - Annual payment comparisons
        
        #### Inflation Linking
        - **Linked loans**: Interest rate increases with inflation each year
        - **Fixed loans**: Interest rate remains constant throughout the loan term
        
        This allows you to model complex mortgage scenarios with mixed rate types.
        """)

if __name__ == "__main__":
    main()