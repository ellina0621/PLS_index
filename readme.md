1.主旨 :本研究複製Fuwei Jiang, Jie Kang, Lingchao Meng (2024)的paper，以PLS建構台灣市場的不確定性指標，並檢測是否能像美國一樣具有顯著預測股價的效果，若有可以作為交易策略的參考指標之一。

2.paper研究動機：作者認為過去文獻建構不確定性指標的作法單一，且他們的做法差異滿大的，還是沒有一個明確的方式，因此提出使用監督式機器學習法來建構一個能同時處理高維度與共線性問題，萃取各 X 變數中最有用的特徵，並提升預測穩定性之不確定性指標。

3.Data介紹
  分為加權指數報酬率 (tej)、跟不確定性各項data(上船已合併資料)
  作者所使用的不確定性指標在網路上都找的到，有些是有研究團隊會固定每月在網站上更新，由於台灣資料沒有那麼健全，我們只能採用美國數據（台灣市場主要受美國影響）。另外，作者有使用vix，台灣也有自己的vix指數，但沒有完整的歷史數據，我們仍採用美國vix_index

  樣本時間：2008/01～2024/12（月頻）

  各項不確定性指標:
  *計量模型
   (1) Financial Uncertainty (FU)  【Ludvigson et al. (2021)】
      以金融變數建構 FAVAR 模型，計算預測誤差的條件變異數，取其平均為總體金融不確定性指標。
   (2) Macroeconomic Uncertainty (MU)  【Jurado et al. (2015)】
      使用數百項月度總體經濟指標，FAVAR 預測誤差的條件變異數平均值代表宏觀不確定性
   (3) economic real Uncertainty (ERU)  【Jurado, Ludvigson, and Ng (2013)】
      使用實體經濟數據（如生產力、需求、政策）

  *問卷調查型
   (1) Consumers Uncertainty (CU) 【Leduc and Liu (2016)】
      方法：密西根大學消費者調查中，回答購車為壞時機之原因若為「未來不確定」，則記為1，以該比例作為不確定性指標
   (2) Survey of Business Uncertainty (SBU)  【Altig et al. (2022)】   
        a. Sales Growth Uncertainty (SGU)　　　　　根據企業對未來12個月銷售成長的預期機率分布
        b. Employment Growth Uncertainty (EGU)　  根據企業對未來12個月雇用成長的預期機率分布

   (3) Professional Forecasters Uncertainty (PFU)【Rossi and Sekhposyan (2015)】
      方法：以 GDP 預測誤差分布的尾端衡量不確定性（季資料轉為月頻，將季值套用於該季三個月）

  *新聞文本型
   (1) Economic Policy Uncertainty (EPU)【Baker et al. (2016)
      資料來源：十大美國主要報紙〔 Wall Street Journal、New York Times、... 〕
      計算方法：具關鍵字的文章數  / 總文章數 後標準化
   (2) Monetary policy uncertainty index (MPU)【Husted et al. (2020)】
   (3) Trade policy uncertainty index (TPU)【Caldara et al. (2020)】
   (4) Geopolitical risk index (GPR)【Caldara and Iacoviello (2022)】

  *市場價格型
   (1) VIX：用Yfinance抓取，並以月平均計算每月資料
   (2) Default Yield Spread (DFY)【Fama and French (1989)】
       方法：BAA 和 AAA 公司債殖利率之間的利差，作為整體信用風險代理（data內取名為credit_spread）

4.實證結果
  (1) pls_index在台灣是否具有預測能力?
      有，且為負向關係。但沒有到真的很顯著，推測不確定性成分跟資料為美國有關。
  (2) 觀察rolling windows下的pls_index預測出下一期的報酬與實際上報酬的差異（圖）
      發現他預測結果跟實際仍有低估及高估的問題，其中低估正報酬及高估負報酬的月份較多。

5.結論
  (1) PLS_index在台灣市場仍具有預測能力，與下期報酬為負相關，建議還是要有台灣數據，預測效果可能會更佳。
  (2) rolling windows所預測的下期報酬率會有估計誤差的問題，仍可以使用此指標作為保守報酬之估計。
