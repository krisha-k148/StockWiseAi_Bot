"""
StockWiseAI Bot - Main Application

Complete frontend with all features integrated

"""

import streamlit as st

import pandas as pd

from datetime import datetime, timedelta



from backend.orchestrator import StockWiseOrchestrator

from backend.utils import format_currency, format_percentage, format_large_number

from config import PAGE_CONFIG, POPULAR_STOCKS, RISK_LEVELS



# Page configuration

st.set_page_config(**PAGE_CONFIG)



# Custom CSS

st.markdown("""

<style>

    .main-header {

        font-size: 3rem;

        font-weight: bold;

        color: #1f77b4;

    }

    .metric-card {

        background-color: #f0f2f6;

        padding: 20px;

        border-radius: 10px;

        border-left: 5px solid #1f77b4;

    }

    .success-box {

        background-color: #d4edda;

        padding: 15px;

        border-radius: 5px;

        border-left: 5px solid #28a745;

    }

    .warning-box {

        background-color: #fff3cd;

        padding: 15px;

        border-radius: 5px;

        border-left: 5px solid #ffc107;

    }

    .danger-box {

        background-color: #f8d7da;

        padding: 15px;

        border-radius: 5px;

        border-left: 5px solid #dc3545;

    }

    .stTabs [data-baseweb="tab-list"] {

        gap: 24px;

    }

    .stTabs [data-baseweb="tab"] {

        height: 50px;

        padding: 10px 20px;

    }

</style>

""", unsafe_allow_html=True)



# Initialize orchestrator

@st.cache_resource

def get_orchestrator():

    return StockWiseOrchestrator()



orchestrator = get_orchestrator()



# Initialize session state

if 'user_id' not in st.session_state:

    st.session_state.user_id = 1



# Main App

def main():

    # Sidebar Navigation

    with st.sidebar:

        st.image("https://via.placeholder.com/150x150.png?text=StockWise", width=150)

        st.title("StockWiseAI Bot")

        st.markdown("---")

        

        page = st.radio(

            "Navigation",

            ["🏠 Home", "💼 Portfolio", "📊 Analysis", "🔮 Forecast", "📚 Education", "⚙️ Settings"],

            label_visibility="collapsed"

        )

        

        st.markdown("---")

        

        # Quick Stats in Sidebar

        st.subheader("Quick Stats")

        portfolio_summary = orchestrator.portfolio_agent.get_portfolio_summary(st.session_state.user_id)

        

        st.metric("Portfolio Value", format_currency(portfolio_summary['total_value']))

        st.metric("Total Gain/Loss", 

                 format_currency(portfolio_summary['total_gain_loss']),

                 f"{portfolio_summary['total_gain_loss_pct']:.2f}%")

        st.metric("Holdings", portfolio_summary['num_stocks'])

        

        st.markdown("---")

        st.caption("StockWiseAI Bot v1.0")

        st.caption("Data powered by Yahoo Finance")

    

    # Route to pages

    if page == "🏠 Home":

        show_home_page()

    elif page == "💼 Portfolio":

        show_portfolio_page()

    elif page == "📊 Analysis":

        show_analysis_page()

    elif page == "🔮 Forecast":

        show_forecast_page()

    elif page == "📚 Education":

        show_education_page()

    elif page == "⚙️ Settings":

        show_settings_page()





def show_home_page():

    """Home/Dashboard Page"""

    st.markdown('<h1 class="main-header">🤖 StockWiseAI Dashboard</h1>', unsafe_allow_html=True)

    st.markdown("### Your AI-Powered Investment Assistant")

    

    # Market Overview

    st.subheader("📈 Market Overview")

    indices = orchestrator.fetcher.get_market_indices()

    

    if indices:

        cols = st.columns(len(indices))

        for col, (name, data) in zip(cols, indices.items()):

            with col:

                change_color = "green" if data['change'] >= 0 else "red"

                st.metric(

                    name,

                    f"${data['current_price']:.2f}",

                    f"{data['change_pct']:.2f}%",

                    delta_color="normal"

                )

    

    st.markdown("---")

    

    # Portfolio Summary

    col1, col2 = st.columns([2, 1])

    

    with col1:

        st.subheader("💼 Portfolio Summary")

        summary = orchestrator.portfolio_agent.get_portfolio_summary(st.session_state.user_id)

        

        if summary['num_stocks'] > 0:

            # Portfolio metrics

            metric_cols = st.columns(3)

            with metric_cols[0]:

                st.metric("Total Value", format_currency(summary['total_value']))

            with metric_cols[1]:

                st.metric("Cost Basis", format_currency(summary['total_cost']))

            with metric_cols[2]:

                st.metric("Total Gain/Loss", 

                         format_currency(summary['total_gain_loss']),

                         f"{summary['total_gain_loss_pct']:.2f}%")

            

            # Top holdings

            st.markdown("#### Top Holdings")

            holdings_df = pd.DataFrame(summary['holdings'])

            holdings_df = holdings_df.sort_values('current_value', ascending=False).head(5)

            

            for _, holding in holdings_df.iterrows():

                col_a, col_b, col_c = st.columns([2, 1, 1])

                with col_a:

                    st.write(f"**{holding['symbol']}**")

                with col_b:

                    st.write(format_currency(holding['current_value']))

                with col_c:

                    color = "🟢" if holding['gain_loss'] >= 0 else "🔴"

                    st.write(f"{color} {holding['gain_loss_pct']:.2f}%")

        else:

            st.info("No stocks in portfolio. Add your first stock to get started!")

            if st.button("Add Stock to Portfolio", type="primary"):

                st.session_state.show_add_stock = True

                st.rerun()

    

    with col2:

        st.subheader("🎯 Quick Actions")

        

        if st.button("📊 Analyze Stock", use_container_width=True):

            st.session_state.quick_action = "analyze"

        

        if st.button("➕ Add to Portfolio", use_container_width=True):

            st.session_state.quick_action = "add"

        

        if st.button("🔮 Get Forecast", use_container_width=True):

            st.session_state.quick_action = "forecast"

        

        if st.button("📚 Start Learning", use_container_width=True):

            st.session_state.quick_action = "learn"

    

    st.markdown("---")

    

    # Quick Stock Lookup

    st.subheader("🔍 Quick Stock Lookup")

    

    col1, col2 = st.columns([3, 1])

    with col1:

        lookup_symbol = st.text_input("Enter stock symbol:", value="", placeholder="e.g., AAPL, MSFT, GOOGL")

    with col2:

        st.write("")

        st.write("")

        lookup_button = st.button("Search", type="primary", use_container_width=True)

    

    if lookup_button and lookup_symbol:

        with st.spinner(f"Fetching data for {lookup_symbol.upper()}..."):

            symbol = lookup_symbol.upper()

            info = orchestrator.fetcher.get_stock_info(symbol)

            current_price = orchestrator.fetcher.get_current_price(symbol)

            

            if info and current_price:

                col_a, col_b = st.columns([1, 2])

                

                with col_a:

                    st.markdown(f"### {symbol}")

                    st.markdown(f"**{info['name']}**")

                    st.metric("Current Price", format_currency(current_price))

                

                with col_b:

                    st.markdown("#### Company Info")

                    st.write(f"**Sector:** {info['sector']}")

                    st.write(f"**Industry:** {info['industry']}")

                    st.write(f"**Market Cap:** {format_large_number(info['market_cap'])}")

                    st.write(f"**P/E Ratio:** {info['pe_ratio']:.2f}" if info['pe_ratio'] else "P/E: N/A")

            else:

                st.error(f"Could not find data for '{symbol}'. Please check the symbol and try again.")

    

    # Popular Stocks

    st.markdown("---")

    st.subheader("🌟 Popular Stocks")

    

    popular_cols = st.columns(5)

    for i, symbol in enumerate(POPULAR_STOCKS[:5]):

        with popular_cols[i]:

            price = orchestrator.fetcher.get_current_price(symbol)

            if price:

                change_info = orchestrator.fetcher.get_price_change(symbol, days=1)

                if change_info:

                    st.metric(

                        symbol,

                        f"${price:.2f}",

                        f"{change_info['change_pct']:.2f}%"

                    )





def show_portfolio_page():

    """Portfolio Management Page"""

    st.title("💼 Portfolio Management")

    

    tabs = st.tabs(["📊 Overview", "➕ Add Stock", "📈 Performance", "⚖️ Rebalance"])

    

    # Tab 1: Overview

    with tabs[0]:

        summary = orchestrator.portfolio_agent.get_portfolio_summary(st.session_state.user_id)

        

        if summary['num_stocks'] == 0:

            st.info("Your portfolio is empty. Add your first stock to get started!")

        else:

            # Portfolio metrics

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric("Total Value", format_currency(summary['total_value']))

            with col2:

                st.metric("Cost Basis", format_currency(summary['total_cost']))

            with col3:

                st.metric("Total Return", format_currency(summary['total_gain_loss']))

            with col4:

                st.metric("Return %", format_percentage(summary['total_gain_loss_pct']))

            

            st.markdown("---")

            

            # Holdings table

            st.subheader("Your Holdings")

            holdings_df = pd.DataFrame(summary['holdings'])

            

            # Format for display

            display_df = holdings_df[[

                'symbol', 'shares', 'purchase_price', 'current_price', 

                'cost_basis', 'current_value', 'gain_loss', 'gain_loss_pct'

            ]].copy()

            

            display_df.columns = [

                'Symbol', 'Shares', 'Buy Price', 'Current Price',

                'Cost Basis', 'Current Value', 'Gain/Loss ($)', 'Gain/Loss (%)'

            ]

            

            # Format currency columns

            for col in ['Buy Price', 'Current Price', 'Cost Basis', 'Current Value', 'Gain/Loss ($)']:

                display_df[col] = display_df[col].apply(lambda x: format_currency(x))

            

            display_df['Gain/Loss (%)'] = display_df['Gain/Loss (%)'].apply(lambda x: format_percentage(x))

            

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            

            # Remove stock option

            st.markdown("---")

            st.subheader("Manage Holdings")

            

            col_a, col_b = st.columns([2, 1])

            with col_a:

                stock_to_remove = st.selectbox(

                    "Select stock to remove:",

                    options=holdings_df['symbol'].tolist()

                )

            with col_b:

                st.write("")

                st.write("")

                if st.button("Remove Stock", type="secondary"):

                    portfolio_id = holdings_df[holdings_df['symbol'] == stock_to_remove]['portfolio_id'].iloc[0]

                    orchestrator.db.remove_from_portfolio(portfolio_id, st.session_state.user_id)

                    st.success(f"Removed {stock_to_remove} from portfolio!")

                    st.rerun()

    

    # Tab 2: Add Stock

    with tabs[1]:

        st.subheader("Add Stock to Portfolio")

        

        col1, col2 = st.columns(2)

        

        with col1:

            new_symbol = st.text_input("Stock Symbol", placeholder="e.g., AAPL").upper()

            new_shares = st.number_input("Number of Shares", min_value=0.01, value=1.0, step=0.01)

            new_price = st.number_input("Purchase Price ($)", min_value=0.01, value=100.0, step=0.01)

        

        with col2:

            new_date = st.date_input("Purchase Date", value=datetime.now())

            new_notes = st.text_area("Notes (optional)", placeholder="Add any notes about this purchase...")

        

        if st.button("Add to Portfolio", type="primary", use_container_width=True):

            if new_symbol and new_shares > 0 and new_price > 0:

                # Validate symbol

                if orchestrator.fetcher.validate_symbol(new_symbol):

                    try:

                        orchestrator.db.add_to_portfolio(

                            user_id=st.session_state.user_id,

                            symbol=new_symbol,

                            shares=new_shares,

                            price=new_price,

                            date=new_date.strftime("%Y-%m-%d"),

                            notes=new_notes

                        )

                        st.success(f"✅ Added {new_shares} shares of {new_symbol} to portfolio!")

                        st.balloons()

                    except Exception as e:

                        st.error(f"Error adding to portfolio: {e}")

                else:

                    st.error(f"Invalid stock symbol: {new_symbol}")

            else:

                st.warning("Please fill in all required fields.")

    

    # Tab 3: Performance

    with tabs[2]:

        st.subheader("Portfolio Performance")

        

        period = st.select_slider(

            "Time Period",

            options=[7, 30, 90, 180, 365],

            value=30,

            format_func=lambda x: f"{x} days"

        )

        

        performance = orchestrator.portfolio_agent.calculate_portfolio_performance(

            st.session_state.user_id, days=period

        )

        

        if 'error' not in performance:

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric("Period Return", format_percentage(performance['total_return_pct']))

            with col2:

                st.metric("Volatility", format_percentage(performance['volatility']))

            with col3:

                st.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")

            

            # Performance chart

            st.markdown("#### Portfolio Value Over Time")

            if 'daily_values' in performance:

                chart_data = pd.DataFrame({

                    'Date': performance['daily_values'].index,

                    'Portfolio Value': performance['daily_values'].values

                })

                st.line_chart(chart_data.set_index('Date'))

        else:

            st.info("Add stocks to your portfolio to see performance metrics.")

    

    # Tab 4: Rebalance

    with tabs[3]:

        st.subheader("Portfolio Rebalancing")

        

        diversification = orchestrator.portfolio_agent.calculate_portfolio_diversification(st.session_state.user_id)

        

        if diversification['sector_allocation']:

            st.markdown("#### Current Sector Allocation")

            

            sector_df = pd.DataFrame({

                'Sector': list(diversification['sector_allocation'].keys()),

                'Allocation (%)': list(diversification['sector_allocation'].values())

            })

            

            st.bar_chart(sector_df.set_index('Sector'))

            

            st.markdown(f"**Concentration Risk:** {diversification['concentration_risk']}")

            st.markdown(f"**Largest Position:** {diversification['largest_position_pct']:.2f}%")

            

            # Rebalancing suggestions

            st.markdown("---")

            st.markdown("#### Rebalancing Suggestions")

            

            suggestions = orchestrator.portfolio_agent.suggest_rebalancing(st.session_state.user_id)

            

            if suggestions['suggestions']:

                for suggestion in suggestions['suggestions']:

                    if suggestion['action'] == 'REDUCE':

                        st.warning(f"🔸 **{suggestion['symbol']}**: {suggestion['reason']}")

                    else:

                        st.info(f"ℹ️ {suggestion['reason']}")

            else:

                st.success("✅ Your portfolio is well-balanced!")

        else:

            st.info("Add stocks to your portfolio to see rebalancing recommendations.")





def show_analysis_page():

    """Stock Analysis Page"""

    st.title("📊 Stock Analysis")

    

    # Stock selection

    col1, col2 = st.columns([3, 1])

    with col1:

        analysis_symbol = st.text_input("Enter stock symbol to analyze:", value="AAPL").upper()

    with col2:

        st.write("")

        st.write("")

        analyze_button = st.button("Analyze", type="primary", use_container_width=True)

    

    if analyze_button or analysis_symbol:

        with st.spinner(f"Analyzing {analysis_symbol}..."):

            # Get comprehensive analysis

            complete_analysis = orchestrator.get_complete_stock_analysis(analysis_symbol)

            

            if complete_analysis['technical'] is None:

                st.error(f"Could not fetch data for {analysis_symbol}")

                return

            

            # Stock header

            info = orchestrator.fetcher.get_stock_info(analysis_symbol)

            if info:

                st.markdown(f"## {info['name']} ({analysis_symbol})")

                st.markdown(f"**Sector:** {info['sector']} | **Industry:** {info['industry']}")

            

            st.markdown("---")

            

            # Tabs for different analysis types

            tabs = st.tabs(["📈 Technical", "💰 Fundamental", "⚠️ Risk", "🔍 Comparison"])

            

            # Technical Analysis Tab

            with tabs[0]:

                technical = complete_analysis['technical']

                

                if technical:

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:

                        st.metric("Current Price", format_currency(technical['current_price']))

                    with col2:

                        st.metric("RSI", f"{technical['rsi']:.2f}")

                    with col3:

                        st.metric("MACD", f"{technical['macd']:.2f}")

                    with col4:

                        signal = technical['signals']['overall']

                        color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}

                        st.metric("Signal", f"{color.get(signal, '⚪')} {signal}")

                    

                    st.markdown("---")

                    

                    # Moving Averages

                    st.markdown("#### Moving Averages")

                    ma_col1, ma_col2, ma_col3 = st.columns(3)

                    with ma_col1:

                        st.metric("MA 20", format_currency(technical['ma_20']))

                    with ma_col2:

                        st.metric("MA 50", format_currency(technical['ma_50']))

                    with ma_col3:

                        st.metric("MA 200", format_currency(technical['ma_200']))

                    

                    # Price chart with indicators

                    st.markdown("#### Price Chart")

                    df = technical['dataframe']

                    

                    chart_data = pd.DataFrame({

                        'Close': df['Close'],

                        'MA 20': df['MA_20'],

                        'MA 50': df['MA_50']

                    })

                    st.line_chart(chart_data)

                    

                    # Signals and recommendations

                    st.markdown("#### Analysis Signals")

                    for detail in technical['signals']['details']:

                        st.info(f"• {detail}")

            

            # Fundamental Analysis Tab

            with tabs[1]:

                fundamental = complete_analysis['fundamental']

                

                if fundamental:

                    # Valuation

                    st.markdown("#### Valuation")

                    val = fundamental['valuation']

                    

                    val_col1, val_col2, val_col3 = st.columns(3)

                    with val_col1:

                        st.metric("P/E Ratio", f"{val['pe_ratio']:.2f}" if val['pe_ratio'] else "N/A")

                    with val_col2:

                        st.metric("Forward P/E", f"{val['forward_pe']:.2f}" if val['forward_pe'] else "N/A")

                    with val_col3:

                        st.metric("Assessment", val['assessment'])

                    

                    if val['notes']:

                        for note in val['notes']:

                            st.write(f"• {note}")

                    

                    st.markdown("---")

                    

                    # Financial Health

                    st.markdown("#### Financial Health")

                    health = fundamental['financial_health']

                    

                    health_col1, health_col2 = st.columns(2)

                    with health_col1:

                        st.metric("Market Cap", format_large_number(health['market_cap']))

                        st.write(f"**Size:** {health['size_category']}")

                    with health_col2:

                        st.metric("Risk Level", health['relative_risk'])

                    

                    st.markdown("---")

                    

                    # Dividend

                    st.markdown("#### Dividend Information")

                    dividend = fundamental['dividend']

                    st.metric("Dividend Yield", format_percentage(dividend['yield']))

                    st.write(f"**Assessment:** {dividend['assessment']}")

            

            # Risk Analysis Tab

            with tabs[2]:

                risk = complete_analysis['risk']

                

                if risk and 'error' not in risk:

                    # Risk score

                    st.markdown(f"### Risk Level: {risk['risk_level']}")

                    st.progress(risk['risk_score'] / 100)

                    

                    risk_col1, risk_col2, risk_col3 = st.columns(3)

                    with risk_col1:

                        st.metric("Risk Score", f"{risk['risk_score']}/100")

                    with risk_col2:

                        st.metric("Volatility", format_percentage(risk['volatility']))

                    with risk_col3:

                        st.metric("Beta", f"{risk['beta']:.2f}")

                    

                    st.markdown("---")

                    

                    # Risk factors

                    st.markdown("#### Risk Factors")

                    for factor in risk['risk_factors']:

                        st.warning(f"• {factor}")

                    

                    st.markdown("---")

                    

                    # Recommendations

                    st.markdown("#### Recommendations")

                    for rec in risk['recommendations']:

                        st.info(rec)

            

            # Comparison Tab

            with tabs[3]:

                st.markdown("#### Compare with Other Stocks")

                

                compare_symbols = st.multiselect(

                    "Select stocks to compare:",

                    options=POPULAR_STOCKS,

                    default=[analysis_symbol] if analysis_symbol in POPULAR_STOCKS else []

                )

                

                if len(compare_symbols) >= 2:

                    comparison = orchestrator.analysis_agent.compare_stocks(compare_symbols)

                    

                    # Performance comparison

                    st.markdown("##### Performance Comparison (1 Year)")

                    perf_df = pd.DataFrame(comparison['performance']).T

                    st.dataframe(perf_df, use_container_width=True)

                    

                    # Metrics comparison

                    st.markdown("##### Metrics Comparison")

                    metrics_df = pd.DataFrame(comparison['metrics']).T

                    st.dataframe(metrics_df, use_container_width=True)





def show_forecast_page():

    """Stock Forecasting Page"""

    st.title("🔮 Stock Price Forecasting")

    

    st.markdown("Use AI and machine learning to predict future stock prices.")

    

    # Stock selection

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:

        forecast_symbol = st.text_input("Enter stock symbol:", value="AAPL").upper()

    with col2:

        forecast_days = st.selectbox("Forecast Period", options=[7, 30, 90], format_func=lambda x: f"{x} days")

    with col3:

        model_type = st.selectbox("Model", options=["rf", "linear", "ensemble"])

    

    if st.button("Generate Forecast", type="primary", use_container_width=True):

        with st.spinner(f"Training model and forecasting {forecast_symbol}..."):

            # Get forecast

            if model_type == "ensemble":

                forecast = orchestrator.forecast_agent.ensemble_forecast(forecast_symbol, days=forecast_days)

            else:

                forecast = orchestrator.forecast_agent.forecast_price(forecast_symbol, days=forecast_days, model_type=model_type)

            

            if forecast and 'error' not in forecast:

                st.success("✅ Forecast generated successfully!")

                

                # Forecast summary

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric("Current Price", format_currency(forecast['current_price']))

                with col2:

                    st.metric("Predicted Price", format_currency(forecast['predicted_price'][-1]))

                with col3:

                    change_pct = forecast['forecast_change_pct']

                    st.metric("Expected Change", format_percentage(change_pct), delta=format_percentage(change_pct))

                

                st.markdown("---")

                

                # Forecast chart

                st.markdown(f"#### {forecast_days}-Day Price Forecast")

                

                forecast_df = pd.DataFrame({

                    'Date': forecast['dates'],

                    'Predicted Price': forecast['predicted_price'],

                    'Lower Bound': forecast.get('lower_bound', forecast['predicted_price']),

                    'Upper Bound': forecast.get('upper_bound', forecast['predicted_price'])

                })

                

                st.line_chart(forecast_df.set_index('Date'))

                

                # Model info

                st.markdown("---")

                st.markdown("#### Model Information")

                st.info(f"**Model Type:** {forecast['model_type']}")

                if 'models_used' in forecast:

                    st.info(f"**Models Combined:** {', '.join(forecast['models_used'])}")

                

                # Disclaimer

                st.warning("⚠️ **Disclaimer:** These predictions are based on historical data and machine learning models. They should not be used as the sole basis for investment decisions. Always do your own research and consult with financial advisors.")

            else:

                st.error("Unable to generate forecast. Please try a different stock or time period.")

    

    st.markdown("---")

    

    # Trend Analysis

    st.subheader("📊 Quick Trend Analysis")

    

    trend_symbol = st.text_input("Stock for trend analysis:", value="MSFT", key="trend_input").upper()

    

    if st.button("Analyze Trend"):

        trend = orchestrator.forecast_agent.get_trend_prediction(trend_symbol)

        

        if 'trend' in trend:

            col1, col2 = st.columns(2)

            

            with col1:

                st.markdown(f"### {trend['trend']}")

                st.markdown(f"**Confidence:** {trend['confidence']}")

                st.metric("Current Price", format_currency(trend['current_price']))

            

            with col2:

                st.metric("10-Day Momentum", format_percentage(trend['momentum_10d']))

                st.metric("30-Day Momentum", format_percentage(trend['momentum_30d']))





def show_education_page():

    """Education and Learning Page"""

    st.title("📚 Investment Education")

    

    tabs = st.tabs(["📖 Tutorials", "📝 Quizzes", "📊 Progress", "📕 Glossary"])

    

    # Tutorials Tab

    with tabs[0]:

        st.subheader("Learning Tutorials")

        

        tutorials = orchestrator.education_agent.get_all_tutorials()

        

        for topic_key, topic_data in tutorials.items():

            with st.expander(f"📘 {topic_data['title']} - {topic_data['difficulty']}"):

                for i, lesson in enumerate(topic_data['lessons'], 1):

                    st.markdown(f"### Lesson {i}: {lesson['title']}")

                    st.markdown(lesson['content'])

                    

                    if lesson['key_terms']:

                        st.markdown("**Key Terms:**")

                        st.write(", ".join(lesson['key_terms']))

                    

                    st.markdown("---")

                    

                    if st.button(f"Mark as Complete", key=f"complete_{topic_key}_{i}"):

                        orchestrator.education_agent.db.update_learning_progress(

                            st.session_state.user_id,

                            f"{topic_key}_lesson_{i}",

                            completed=True

                        )

                        st.success("✅ Lesson marked as complete!")

    

    # Quizzes Tab

    with tabs[1]:

        st.subheader("Test Your Knowledge")

        

        available_quizzes = orchestrator.education_agent.get_available_quizzes()

        

        quiz_options = {quiz['title']: quiz['id'] for quiz in available_quizzes}

        selected_quiz_title = st.selectbox("Select a quiz:", options=list(quiz_options.keys()))

        selected_quiz_id = quiz_options[selected_quiz_title]

        

        if st.button("Start Quiz", type="primary"):

            st.session_state.active_quiz = selected_quiz_id

            st.session_state.quiz_answers = []

            st.session_state.quiz_question_index = 0

            st.rerun()

        

        # Display quiz if active

        if 'active_quiz' in st.session_state and st.session_state.active_quiz:

            quiz = orchestrator.education_agent.get_quiz(st.session_state.active_quiz)

            

            if quiz:

                question_index = st.session_state.quiz_question_index

                questions = quiz['questions']

                

                if question_index < len(questions):

                    question = questions[question_index]

                    st.markdown(f"### Question {question_index + 1} of {len(questions)}")

                    st.markdown(f"**{question['question']}**")

                    

                    # Display options

                    selected_answer = st.radio(

                        "Select your answer:",

                        options=question['options'],

                        key=f"quiz_answer_{question_index}"

                    )

                    

                    col1, col2 = st.columns([1, 1])

                    with col1:

                        if st.button("Previous", disabled=(question_index == 0)):

                            st.session_state.quiz_question_index -= 1

                            st.rerun()

                    

                    with col2:

                        if st.button("Next", type="primary"):

                            # Save answer

                            st.session_state.quiz_answers.append({

                                'question_id': question['id'],

                                'answer': selected_answer

                            })

                            

                            if question_index < len(questions) - 1:

                                st.session_state.quiz_question_index += 1

                                st.rerun()

                            else:

                                # Quiz complete - calculate score

                                score = orchestrator.education_agent.submit_quiz(

                                    st.session_state.user_id,

                                    st.session_state.active_quiz,

                                    st.session_state.quiz_answers

                                )

                                
                                st.success(f"✅ Quiz completed! Your score: {score['score']}/{score['total']} ({score['percentage']:.1f}%)")

                                
                                del st.session_state.active_quiz

                                del st.session_state.quiz_answers

                                del st.session_state.quiz_question_index

                                st.rerun()

                else:

                    st.info("Quiz completed!")

    

    # Progress Tab

    with tabs[2]:

        st.subheader("Your Learning Progress")

        

        progress = orchestrator.education_agent.get_user_progress(st.session_state.user_id)

        

        if progress:

            # Overall progress

            total_lessons = progress.get('total_lessons', 0)

            completed_lessons = progress.get('completed_lessons', 0)

            progress_pct = (completed_lessons / total_lessons * 100) if total_lessons > 0 else 0

            

            st.metric("Overall Progress", f"{completed_lessons}/{total_lessons} lessons", f"{progress_pct:.1f}%")

            st.progress(progress_pct / 100)

            

            st.markdown("---")

            

            # Topic progress

            st.markdown("#### Progress by Topic")

            if 'topic_progress' in progress:

                for topic, topic_data in progress['topic_progress'].items():

                    topic_completed = topic_data.get('completed', 0)

                    topic_total = topic_data.get('total', 0)

                    topic_pct = (topic_completed / topic_total * 100) if topic_total > 0 else 0

                    

                    st.write(f"**{topic_data.get('title', topic)}**")

                    st.progress(topic_pct / 100)

                    st.write(f"{topic_completed}/{topic_total} lessons completed")

                    st.markdown("---")

            

            # Quiz scores

            st.markdown("#### Quiz Scores")

            if 'quiz_scores' in progress and progress['quiz_scores']:

                quiz_df = pd.DataFrame(progress['quiz_scores'])

                st.dataframe(quiz_df, use_container_width=True, hide_index=True)

            else:

                st.info("No quiz scores yet. Complete a quiz to see your scores here!")

        else:

            st.info("Start learning to track your progress!")

    

    # Glossary Tab

    with tabs[3]:

        st.subheader("Investment Glossary")

        

        glossary = orchestrator.education_agent.get_glossary()

        

        search_term = st.text_input("Search term:", placeholder="e.g., P/E Ratio, Dividend...")

        

        if search_term:

            filtered_glossary = {

                k: v for k, v in glossary.items()

                if search_term.lower() in k.lower() or search_term.lower() in v.get('definition', '').lower()

            }

        else:

            filtered_glossary = glossary

        

        for term, data in filtered_glossary.items():

            with st.expander(f"**{term}**"):

                st.markdown(data.get('definition', 'No definition available'))

                if data.get('example'):

                    st.markdown(f"*Example:* {data['example']}")

                if data.get('related_terms'):

                    st.markdown(f"*Related:* {', '.join(data['related_terms'])}")





def show_settings_page():

    """Settings and Preferences Page"""

    st.title("⚙️ Settings")

    

    tabs = st.tabs(["👤 Profile", "🔔 Notifications", "📊 Preferences", "🔒 Privacy"])

    

    # Profile Tab

    with tabs[0]:

        st.subheader("User Profile")

        

        col1, col2 = st.columns(2)

        

        with col1:

            user_name = st.text_input("Name", value=st.session_state.get('user_name', 'User'))

            user_email = st.text_input("Email", value=st.session_state.get('user_email', 'user@example.com'))

            risk_tolerance = st.selectbox(

                "Risk Tolerance",

                options=list(RISK_LEVELS.keys()),

                index=0

            )

        

        with col2:

            investment_goal = st.selectbox(

                "Investment Goal",

                options=["Growth", "Income", "Balanced", "Preservation"],

                index=0

            )

            investment_horizon = st.selectbox(

                "Investment Horizon",

                options=["Short-term (< 1 year)", "Medium-term (1-5 years)", "Long-term (5+ years)"],

                index=2

            )

            currency = st.selectbox(

                "Currency",

                options=["USD", "EUR", "GBP", "JPY", "CAD"],

                index=0

            )

        

        if st.button("Save Profile", type="primary"):

            st.session_state.user_name = user_name

            st.session_state.user_email = user_email

            st.session_state.risk_tolerance = risk_tolerance

            st.session_state.investment_goal = investment_goal

            st.session_state.investment_horizon = investment_horizon

            st.session_state.currency = currency

            st.success("✅ Profile settings saved!")

    

    # Notifications Tab

    with tabs[1]:

        st.subheader("Notification Preferences")

        

        st.markdown("Configure how you want to be notified about your portfolio and market updates.")

        

        email_notifications = st.checkbox("Email Notifications", value=True)

        price_alerts = st.checkbox("Price Alerts", value=True)

        portfolio_updates = st.checkbox("Portfolio Updates", value=True)

        market_news = st.checkbox("Market News", value=False)

        weekly_summary = st.checkbox("Weekly Summary", value=True)

        

        if price_alerts:

            alert_threshold = st.slider(

                "Price Alert Threshold (%)",

                min_value=1.0,

                max_value=20.0,

                value=5.0,

                step=0.5

            )

        

        if st.button("Save Notification Settings", type="primary"):

            st.session_state.email_notifications = email_notifications

            st.session_state.price_alerts = price_alerts

            st.session_state.portfolio_updates = portfolio_updates

            st.session_state.market_news = market_news

            st.session_state.weekly_summary = weekly_summary

            if price_alerts:

                st.session_state.alert_threshold = alert_threshold

            st.success("✅ Notification settings saved!")

    

    # Preferences Tab

    with tabs[2]:

        st.subheader("Display Preferences")

        

        col1, col2 = st.columns(2)

        

        with col1:

            theme = st.selectbox(

                "Theme",

                options=["Light", "Dark", "Auto"],

                index=0

            )

            date_format = st.selectbox(

                "Date Format",

                options=["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"],

                index=0

            )

            number_format = st.selectbox(

                "Number Format",

                options=["US (1,234.56)", "European (1.234,56)", "Indian (12,34,567.89)"],

                index=0

            )

        

        with col2:

            default_timeframe = st.selectbox(

                "Default Timeframe",

                options=["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "5 Years"],

                index=2

            )

            chart_type = st.selectbox(

                "Default Chart Type",

                options=["Line", "Candlestick", "Area"],

                index=0

            )

            refresh_interval = st.selectbox(

                "Auto-refresh Interval",

                options=["Never", "30 seconds", "1 minute", "5 minutes", "15 minutes"],

                index=2

            )

        

        st.markdown("---")

        st.subheader("Analysis Preferences")

        

        default_model = st.selectbox(

            "Default Forecast Model",

            options=["Random Forest", "Linear Regression", "Ensemble"],

            index=2

        )

        technical_indicators = st.multiselect(

            "Default Technical Indicators",

            options=["RSI", "MACD", "Bollinger Bands", "Moving Averages", "Volume"],

            default=["RSI", "MACD", "Moving Averages"]

        )

        

        if st.button("Save Preferences", type="primary"):

            st.session_state.theme = theme

            st.session_state.date_format = date_format

            st.session_state.number_format = number_format

            st.session_state.default_timeframe = default_timeframe

            st.session_state.chart_type = chart_type

            st.session_state.refresh_interval = refresh_interval

            st.session_state.default_model = default_model

            st.session_state.technical_indicators = technical_indicators

            st.success("✅ Preferences saved!")

    

    # Privacy Tab

    with tabs[3]:

        st.subheader("Privacy & Security")

        

        st.markdown("### Data Management")

        

        st.info("""

        **Data Storage:** Your portfolio data is stored locally in your browser session.

        No personal financial data is transmitted to external servers.

        """)

        

        if st.button("Clear All Data", type="secondary"):

            if st.checkbox("I understand this will delete all my portfolio data", key="confirm_clear"):

                # Clear portfolio data

                orchestrator.db.clear_user_portfolio(st.session_state.user_id)

                st.session_state.clear()

                st.session_state.user_id = 1

                st.warning("⚠️ All data has been cleared. Please refresh the page.")

                st.rerun()

        

        st.markdown("---")

        st.markdown("### Export Data")

        

        if st.button("Export Portfolio Data", type="primary"):

            portfolio_data = orchestrator.portfolio_agent.get_portfolio_summary(st.session_state.user_id)

            holdings_df = pd.DataFrame(portfolio_data['holdings'])

            csv = holdings_df.to_csv(index=False)

            st.download_button(

                label="Download CSV",

                data=csv,

                file_name=f"portfolio_export_{datetime.now().strftime('%Y%m%d')}.csv",

                mime="text/csv"

            )

        

        st.markdown("---")

        st.markdown("### About")

        st.info("""

        **StockWiseAI Bot v1.0**

        - Powered by Yahoo Finance API

        - Built with Streamlit

        - Machine Learning forecasting models

        - Educational content for investors

        """)



if __name__ == "__main__":

    main()

