 一:测试集中有重复的USER_ID, COUPON_ID, DATE_RECEIVED，怎么测评
答：三者都重复只计算一个，提交分数取max那个，真实label则取（1如果有至少一个被使用了）

二:为什么提交的结果AUC=0.5的原因说明
答：1.没有严格按照user_id,coupon_id,date_received,probability这样4列提交。特别地，第二列是coupon_id不是merchant_id，第二列是coupon_id不是merchant_id，第二列是coupon_id不是merchant_id；
2.分隔符使用了中文'，'而不是英文','；
3.提交的user_id,coupon_id,date_received与测试集不同（比如直接提交sample_submission.csv）
4. 预测结果的所有概率值相等

###綫上COUPON 與線下  無交集##########


User Item
Train Online User (762858, 2)
Train Offline User (539438, 2)
Test User (76309, 1)

test item 112803
Action   0 点击， 1购买，2领取优惠券

測試綫上  與測試線下  交集很少

測試綫上  點擊 》》》購買》》領券
測試綫上  Merchant_id  7999
測試綫上  fixed 都會即刻使用


商戶
點擊  購買 領券 數量

SAMPLE uSER 209

       User_id  Merchant_id  Coupon_id Discount_rate Distance  Date_received
23866      209         5032       7557          20:5        1       20160721
23867      209         5032        825          20:5        1       20160721
         User_id  Merchant_id Coupon_id Discount_rate Distance Date_received  \
1265423      209         3267      null          null        3          null

             Date
1265423  20160605
         User_id  Merchant_id  Action  Coupon_id Discount_rate Date_received  \
3609252      209        27710       1       null          null          null
3609253      209        27710       1       null          null          null
3609254      209        27710       1  100081876        150:10      20160514
3609255      209        35500       0       null          null          null
3609256      209        27710       1       null          null          null

             Date
3609252  20160514
3609253  20160514
3609254  20160514
3609255  20160123
3609256  20160514