# 增强版动态持仓管理策略 v8.2 (因子化/最终修复版)
# 导入聚宽函数库
import jqdata
import pandas as pd
from jqdata import *
import datetime
import numpy as np

# --- 新增：导入配置和jqdatasdk ---
try:
    from config import get_jqdata_auth
    import jqdatasdk as jqd
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
# --- 结束新增 ---

# 创建logger避免与numpy.log冲突
class Logger:
    def info(self, msg): 
        # 生产环境可考虑降低日志频率或使用 debug
        print("INFO: {}".format(msg)) 
    def debug(self, msg): 
        # 用于更详细的调试信息
        # print("DEBUG: {}".format(msg))
        pass
    def warning(self, msg): print("WARNING: {}".format(msg))
    def error(self, msg): print("ERROR: {}".format(msg))

# 在研究环境中使用自定义logger
if hasattr(np.log, '__call__') and not hasattr(np.log, 'info'):
    logger = Logger()
    log = logger
else:
    logger = log

# 导入技术指标库
try:
    from jqfactor import get_factor_values
    import talib
    FACTOR_API_AVAILABLE = True
    print("✅ JQFactor和TA-Lib库导入成功")
except ImportError:
    FACTOR_API_AVAILABLE = False
    print("❌ JQFactor或TA-Lib库导入失败")

def initialize(context):
    """策略初始化"""
    # --- 新增：JQData SDK 认证 ---
    if CONFIG_AVAILABLE:
        try:
            username, password = get_jqdata_auth()
            jqd.auth(username, password)
            print("✅ JQData SDK 认证成功")
        except Exception as e:
            print("❌ JQData SDK 认证失败: {}".format(e))
    # --- 结束新增 ---
    
    set_option('use_real_price', True)
    set_option('order_volume_ratio', 0.25)
    set_option('avoid_future_data', True)
    set_benchmark('000300.XSHG')
    
    # 使用 context.params 存储策略参数，便于管理和调参
    context.params = {
        'daily_buy_count': 5,
        'atr_stop_multiplier': 3.0,
        'cooldown_days': 5,
        'min_market_cap': 20, # 亿元
        'rsi_lower_bound': 30,
        'rsi_upper_bound': 75,
        'cci_lower_bound': -150,
        'cci_upper_bound': 150,
        'min_turnover_ratio': 1.0, # %
        'boll_score_weight': 0.6,
        'pe_score_weight': 0.4,
        # 用于动态ATR乘数的参数
        'atr_multiplier_low_volatility': 2.5,   # 低波动率市场下的ATR乘数
        'atr_multiplier_high_volatility': 3.5,  # 高波动率市场下的ATR乘数
        'atr_multiplier_threshold': 0.015,      # 波动率判断阈值 (例如，14日ATR的MA20的1.5%)
    }

    # 初始化全局变量
    g.position_entry_info = {}
    g.cooldown_until = {}
    # 初始化缓存
    g.hist_atr_cache = pd.DataFrame()
    g.hist_ma_cache = pd.DataFrame()
    # 用于存储动态ATR乘数的缓存
    g.dynamic_atr_multipliers = {}

    log.info("🔥 === 动态持仓管理策略 v8.2 (因子化/最终修复版) 初始化完成 ===")
    
    # 每日计划
    run_daily(before_trading_start, '08:45')
    run_daily(filter_and_rank_stocks, '09:00')
    run_daily(execute_buys, '09:31')
    run_daily(daily_summary, '15:01')

def before_trading_start(context):
    """盘前准备 - 重置每日变量"""
    g.filtered_stocks = []
    g.today_bought_stocks = set()
    g.today_sold_stocks = []
    today = context.current_dt.date()
    g.cooldown_until = {stock: end_date for stock, end_date in g.cooldown_until.items() if end_date > today}

    # 为handle_data预加载历史数据，提升性能
    # 获取当前持仓列表
    position_list = list(context.portfolio.positions.keys())
    if position_list:
        # 预加载15日历史数据用于ATR计算
        g.hist_atr_cache = get_price(position_list, end_date=context.previous_date, frequency='1d', fields=['high', 'low', 'close'], count=15, panel=False)
        # 预加载10日历史数据用于MA计算
        g.hist_ma_cache = get_price(position_list, end_date=context.previous_date, frequency='1d', fields=['close'], count=10, panel=False)
        
        # --- 新增：计算并缓存动态ATR乘数 ---
        g.dynamic_atr_multipliers = {}
        for security in position_list:
            security_hist_atr = g.hist_atr_cache[g.hist_atr_cache['code'] == security]
            if len(security_hist_atr) >= 14:
                try:
                    atr_values = talib.ATR(security_hist_atr['high'].values, security_hist_atr['low'].values, security_hist_atr['close'].values, timeperiod=14)
                    # 计算ATR的20日移动平均，作为市场波动率基准
                    if len(atr_values) >= 20:
                        atr_ma20 = talib.MA(atr_values, timeperiod=20)
                        current_atr = atr_values[-1]
                        current_atr_ma20 = atr_ma20[-1]
                        
                        # 如果当前ATR相对于其MA20的比率超过阈值，则认为是高波动率市场
                        if current_atr_ma20 > 0 and (current_atr / current_atr_ma20) > context.params['atr_multiplier_threshold']:
                            g.dynamic_atr_multipliers[security] = context.params['atr_multiplier_high_volatility']
                        else:
                            g.dynamic_atr_multipliers[security] = context.params['atr_multiplier_low_volatility']
                    else:
                        # 数据不足时，使用默认乘数
                        g.dynamic_atr_multipliers[security] = context.params['atr_stop_multiplier']
                except Exception as e:
                    log.warning("计算 {} 的动态ATR乘数时出错: {}".format(security, e))
                    g.dynamic_atr_multipliers[security] = context.params['atr_stop_multiplier']
            else:
                # 数据不足时，使用默认乘数
                g.dynamic_atr_multipliers[security] = context.params['atr_stop_multiplier']
        # --- 结束新增 ---
    else:
        g.hist_atr_cache = pd.DataFrame()
        g.hist_ma_cache = pd.DataFrame()
        g.dynamic_atr_multipliers = {} # 初始化为空字典

def get_all_a_shares(date):
    """获取全A股列表，并进行基础过滤"""
    all_stocks = get_all_securities(types=['stock'], date=date).index.tolist()
    # 基础过滤：剔除科创板(68)、北交所(8,4)
    all_stocks = [s for s in all_stocks if not s.startswith('68') and not s.startswith('8') and not s.startswith('4')]
    # 剕除上市不足60天的股票
    all_stocks = [s for s in all_stocks if (date - get_security_info(s).start_date).days > 60]
    
    # 使用 get_extras 获取ST状态，这是支持批量操作的正确方法
    try:
        st_map = get_extras('is_st', all_stocks, start_date=date, end_date=date, df=True)
        non_st_stocks = st_map.iloc[0][st_map.iloc[0] == False].index.tolist()
        return non_st_stocks
    except Exception as e:
        log.error("获取ST状态失败: {}".format(e))
        return all_stocks

def filter_and_rank_stocks(context):
    """使用JQFactor和TA-Lib进行一体化的筛选和评分"""
    if not FACTOR_API_AVAILABLE:
        log.error("JQFactor不可用，无法执行选股。")
        return

    date = context.previous_date
    stock_universe = get_all_a_shares(date)
    log.info("初始股票池（全A股-过滤后）: {} 只".format(len(stock_universe)))

    # --- 新增：调试代码，打印部分可用因子 ---
    # 注意：在生产环境中应禁用此段代码，因为它会打印大量信息
    # 仅在本地或研究环境中用于调试因子名称
    """
    try:
        all_factors = get_all_factors()
        log.info("所有可用因子数量: {}".format(len(all_factors)))
        # 打印包含特定关键词的因子
        pe_factors = all_factors[all_factors['factor'].str.contains('pe', case=False, na=False)]
        log.info("包含'pe'的因子前10个: {}".format(list(pe_factors['factor'][:10])))
        turnover_factors = all_factors[all_factors['factor'].str.contains('turnover', case=False, na=False)]
        log.info("包含'turnover'的因子: {}".format(list(turnover_factors['factor'])))
        market_cap_factors = all_factors[all_factors['factor'].str.contains('market_cap', case=False, na=False)]
        log.info("包含'market_cap'的因子: {}".format(list(market_cap_factors['factor'])))
    except Exception as e:
        log.error("查询所有因子时出错: {}".format(e))
    """
    # --- 结束新增 ---
    
    # --- 修改：再次调整因子名称，使用更可能被识别的名称 ---
    # 根据 yinzi.py 和运行时反馈，调整因子名
    # 市盈率尝试使用 pe_ratio
    # 换手率使用 VOL20
    # 流通市值使用 circulating_market_cap
    fundamental_factors = ['pe_ratio', 'VOL20', 'circulating_market_cap']
    # --- 结束修改 ---
    try:
        fundamental_factor_data = get_factor_values(stock_universe, fundamental_factors, end_date=date, count=1)
        log.info("使用调整后的因子名称获取数据成功: {}".format(fundamental_factors))
    except Exception as e:
        log.error("使用调整后的因子名称获取基本面因子数据失败: {}".format(e))
        return

    # 获取价格数据用于技术指标计算
    try:
        prices_df = get_price(stock_universe, end_date=date, frequency='daily', fields=['open', 'high', 'low', 'close'], count=20, panel=False)
        if prices_df.empty:
            log.warning("未能获取任何价格数据。")
            return
    except Exception as e:
        log.error("获取价格数据失败: {}".format(e))
        return

    # 使用价格数据和talib计算技术指标
    tech_indicators = {}
    unique_stocks = prices_df['code'].unique()
    for security in unique_stocks:
        security_prices = prices_df[prices_df['code'] == security].sort_values('time')
        if len(security_prices) < 14: # 至少需要14天数据计算RSI/CCI
            continue
            
        close_prices = security_prices['close'].values
        high_prices = security_prices['high'].values
        low_prices = security_prices['low'].values
        
        # 计算技术指标
        try:
            rsi = talib.RSI(close_prices, timeperiod=14)[-1] # 取最新值
            cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)[-1] # 取最新值
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
            boll_up = upper[-1]
            boll_down = lower[-1]
            
            tech_indicators[security] = {
                'RSI': rsi,
                'CCI': cci,
                'boll_up': boll_up,
                'boll_down': boll_down,
                'close': close_prices[-1] # 最新收盘价
            }
        except Exception as e:
            log.warning("计算 {} 技术指标时出错: {}".format(security, e))
            continue

    # 合并基本面和技 术指标数据
    df = pd.DataFrame(index=unique_stocks)
    # 填充基本面因子
    for factor in fundamental_factors:
        df[factor] = fundamental_factor_data[factor].iloc[0]
    # 填充技术指标
    for security in unique_stocks:
        if security in tech_indicators:
            for key, value in tech_indicators[security].items():
                df.loc[security, key] = value
            
    df.dropna(inplace=True)
    log.info("获取数据并去NaN后剩余: {} 只".format(len(df)))

    # 使用 context.params 中的参数
    # 现在我们直接使用了调整后的标准因子名，可以直接引用
    df = df[df['circulating_market_cap'] > context.params['min_market_cap']]
    df = df[(df['RSI'] >= context.params['rsi_lower_bound']) & (df['RSI'] <= context.params['rsi_upper_bound'])]
    df = df[(df['CCI'] >= context.params['cci_lower_bound']) & (df['CCI'] <= context.params['cci_upper_bound'])]
    # 使用标准因子名 'VOL20' 进行换手率筛选
    # 注意：VOL20 是百分比形式（如 2.5 表示 2.5%），而 context.params['min_turnover_ratio'] 也是以 % 为单位
    df = df[df['VOL20'] > context.params['min_turnover_ratio']]
    log.info("使用换手率因子: VOL20")
    log.info("换手率筛选后剩余: {} 只".format(len(df)))

    # 因子稳健性处理: 对PE等因子进行异常值过滤
    # 使用调整后的标准因子名 'pe_ratio' 进行PE筛选
    df = df[(df['pe_ratio'] > 0) & (df['pe_ratio'] < 200)]
    log.info("使用市盈率因子: pe_ratio")
    log.info("PE筛选后剩余: {} 只".format(len(df)))

    denominator = df['boll_up'] - df['boll_down']
    df['boll_position'] = (df['close'] - df['boll_down']) / denominator
    df.loc[denominator == 0, 'boll_position'] = np.nan # 处理除零问题
    df.dropna(inplace=True)

    # 使用调整后的标准因子名 'pe_ratio' 计算PE得分
    df['pe_score'] = df['pe_ratio'].rank(pct=True)
    # 使用 context.params 中的权重
    df['final_score'] = context.params['boll_score_weight'] * df['boll_position'] + context.params['pe_score_weight'] * (1 - df['pe_score'])
    
    df = df.sort_values(by='final_score', ascending=False)
    g.filtered_stocks = df.index[:20].tolist()
    log.info("✅ 最终选股与评分完成，选出: {} 只，首选: {}".format(len(g.filtered_stocks), g.filtered_stocks[0] if g.filtered_stocks else '无'))

def execute_buys(context):
    """执行买入操作"""
    today = context.current_dt.weekday()
    if today in [1, 4]:
        log.info("交易日过滤：今日是周{}，不执行买入。".format(today+1))
        return

    # 使用 context.params 中的参数
    # 大盘趋势过滤
    index_hist = attribute_history('000300.XSHG', 20, '1d', ['close'], df=False)
    if len(index_hist['close']) >= 20:
        index_ma20 = index_hist['close'][-20:].mean()
        current_index_price = index_hist['close'][-1]
        if current_index_price < index_ma20:
            log.info("大盘趋势过滤：沪深300指数 {:.2f} < 20日均线 {:.2f}，暂停买入。".format(current_index_price, index_ma20))
            return
    # --- 结束修改 ---

    if not g.filtered_stocks: 
        log.info("今日无筛选出的股票可买入。")
        return
    
    stocks_to_buy = [s for s in g.filtered_stocks if s not in context.portfolio.positions and s not in g.cooldown_until][:context.params['daily_buy_count']] # 使用 params
    if not stocks_to_buy: 
        log.info("筛选出的股票均已持仓或在冷却期，无新股买入。")
        return

    log.info("准备买入 {} 只评分最高的股票".format(len(stocks_to_buy)))
    cash_per_stock = context.portfolio.total_value * 0.1
    target_value = min(cash_per_stock, 200000)

    current_data = get_current_data()
    for security in stocks_to_buy:
        if context.portfolio.available_cash < target_value * 0.9: break
        current_price = current_data[security].last_price
        if current_price is None or current_price <= 0: continue
        amount = int(target_value / current_price // 100) * 100
        if amount < 100: continue
        
        # --- 新增：流动性冲击检查 ---
        # 检查当日成交量，避免吃掉过多流动性
        volume = current_data[security].volume
        if volume == 0: 
            log.debug("跳过 {}: 当日无成交量".format(security))
            continue
        available_volume = volume * 0.25 # 最大吃掉25%成交量
        amount = min(amount, available_volume // 100 * 100)
        if amount < 100: 
            log.debug("跳过 {}: 可用成交量不足1手".format(security))
            continue
        # --- 结束新增 ---
        
        order(security, amount)
        g.today_bought_stocks.add(security)
        g.position_entry_info[security] = {
            'entry_date': context.current_dt.date(),
            'highest_price': max(context.portfolio.positions[security].avg_cost, current_price) # 与handle_data保持一致
        }

def handle_data(context, data):
    """每分钟运行的持仓管理逻辑"""
    # 临近收盘时间限制交易，避免在集合竞价阶段下单
    current_time = context.current_dt.time()
    if current_time >= datetime.time(14, 55):
        log.info("临近收盘时间 {}，暂停执行卖出操作。".format(current_time))
        return

    if not context.portfolio.positions: return

    position_list = list(context.portfolio.positions.keys())
    # 使用在before_trading_start中预加载的数据
    hist_atr = g.hist_atr_cache
    hist_ma = g.hist_ma_cache

    for security in position_list:
        if security in g.today_bought_stocks: continue
        if data[security].paused: continue

        position = context.portfolio.positions[security]
        current_price = data[security].close

        # 单票最大持仓比例检查
        if position.value / context.portfolio.total_value > 0.15: # 单票不超过15%
            log.debug("持仓比例检查: {} 当前持仓比例 {:.2%} 超过15%，跳过卖出检查。".format(security, position.value / context.portfolio.total_value))
            continue

        # 修复：初始化 highest_price 为 max(成本价, 当前价)，避免初始为0导致的问题
        if security not in g.position_entry_info: 
            g.position_entry_info[security] = {
                'entry_date': context.current_dt.date(), 
                'highest_price': max(position.avg_cost, current_price)
            }
        else:
            # 仅在已存在时更新最高价，避免覆盖初始化逻辑
            g.position_entry_info[security]['highest_price'] = max(g.position_entry_info[security]['highest_price'], current_price)

        ma10_series = hist_ma[hist_ma['code'] == security]['close']
        if len(ma10_series) >= 10:
            ma10 = ma10_series.iloc[-10:].mean() # 修复：明确计算最近10日的移动平均
        else:
            log.warning("无法计算 {} 的10日均线，数据不足。".format(security))
            continue
            
        if current_price < ma10:
            log.info("触发趋势止损: {}, 现价{:.2f} < 10日均线{:.2f}".format(security, current_price, ma10))
            order_target(security, 0)
            g.today_sold_stocks.append(security)
            g.cooldown_until[security] = context.current_dt.date() + datetime.timedelta(days=context.params['cooldown_days']) # 使用 params
            if security in g.position_entry_info: del g.position_entry_info[security]
            continue

        try:
            security_hist_atr = hist_atr[hist_atr['code'] == security]
            if len(security_hist_atr) >= 14:
                atr = talib.ATR(security_hist_atr['high'].values, security_hist_atr['low'].values, security_hist_atr['close'].values, timeperiod=14)[-1]
                # --- 修改：使用动态ATR乘数 ---
                dynamic_multiplier = g.dynamic_atr_multipliers.get(security, context.params['atr_stop_multiplier'])
                stop_price = g.position_entry_info[security]['highest_price'] - dynamic_multiplier * atr
                # --- 结束修改 ---
                if current_price < stop_price:
                    log.info("触发ATR止损: {}, 当前价:{:.2f} < 止损线:{:.2f} (动态乘数: {:.2f})".format(security, current_price, stop_price, dynamic_multiplier))
                    order_target(security, 0)
                    g.today_sold_stocks.append(security)
                    g.cooldown_until[security] = context.current_dt.date() + datetime.timedelta(days=context.params['cooldown_days']) # 使用 params
                    if security in g.position_entry_info: del g.position_entry_info[security]
                    continue
        except Exception as e:
            log.warning("计算 {} 的ATR止损时出错: {}".format(security, e))
            pass

def daily_summary(context):
    """每日收盘后总结"""
    log.info("=== 今日收盘总结 ===")
    if not g.today_sold_stocks and not g.today_bought_stocks:
        log.info("本日无任何交易。")
    else:
        if g.today_sold_stocks:
            log.info("本日卖出 {} 只股票: {}".format(len(g.today_sold_stocks), g.today_sold_stocks))
        if g.today_bought_stocks:
            log.info("本日买入 {} 只股票: {}".format(len(g.today_bought_stocks), list(g.today_bought_stocks)))
    log.info("====================")
