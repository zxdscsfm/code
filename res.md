# 椤圭洰缁撴瀯鎬荤粨

## 1. 杩欐槸涓€涓粈涔堜换鍔?
浠庝唬鐮佷富绾跨湅锛岃繖涓」鐩槸涓€涓?*鍩轰簬 Flower 鐨勮仈閭﹀急鐩戠潱鍖诲鍥惧儚鍒嗗壊**椤圭洰锛屾牳蹇冪洰鏍囦笉鏄櫘閫氬崟鏈哄垎鍓诧紝鑰屾槸锛?
- 澶氬鎴风/澶氬煙鏁版嵁涓婂仛鑱旈偊璁粌锛?- 姣忎釜瀹㈡埛绔彲浠ヤ娇鐢ㄤ笉鍚岀被鍨嬬殑寮辨爣娉ㄧ洃鐫ｏ紝渚嬪 `mask`銆乣scribble`銆乣scribble_noisy`銆乣keypoint`銆乣block`銆乣box`锛?- 閫氳繃涓€у寲鍙傛暟鍜?prompt 鏈哄埗澶勭悊璺ㄦ満鏋勫垎甯冨樊寮傦紱
- 褰撳墠涓诲疄楠岀増鏈槸 `FedLPPA`锛岄粯璁ょ粍鍚堟槸 `FedUniV2.1 + UNet_UniV5`銆?
浠庝唬鐮佺粏鑺傚彲浠ョ洿鎺ョ湅鍑鸿繖涓€鐐癸細

- `code_v4/flower_pCE_2D_v4_FedLPPA.py` 鍚屾椂鏀寔 `server/client` 涓ょ瑙掕壊锛屽苟閫氳繃 Flower 鍚姩鑱旈偊璁粌锛?- `code_v4/dataloaders/dataset.py` 鎸?`client1~client5/6` 鍜?`Domain1~Domain6` 鍒掑垎鏁版嵁锛?- `code_v4/networks/unet.py` 鐨?`UNet_UniV5` 鏄庣‘鍖呭惈 `distribution_prompts`銆乣uni_prompt`銆乣label_prompt`锛?- `code_v4/flower_common_v4.py` 涓湁 `FedAvg/FedProx/FedAP/FedAN/FedRep/FedUni/FedUniV2` 绛夎仈閭︾瓥鐣ュ疄鐜帮紱
- 璁粌鏃朵笉浠呮湁鍒嗗壊 CE loss锛岃繕鏈?prompt 涓€鑷存€с€佷吉鏍囩鍙屽垎鏀害鏉熴€佷釜鎬у寲鑱氬悎绛夐€昏緫銆?
涓€鍙ヨ瘽姒傛嫭锛氳繖鏄竴涓?*闈㈠悜澶氫腑蹇冨尰瀛﹀浘鍍忋€佸紓鏋勫急鏍囨敞鍦烘櫙鐨勪釜鎬у寲鑱旈偊鍒嗗壊妗嗘灦**銆?
## 2. 鏁翠綋鐩綍缁撴瀯

- `code_v4/`
  - 椤圭洰涓讳唬鐮佺洰褰曪紝鍖呭惈璁粌鍏ュ彛銆佽仈閭︾瓥鐣ャ€佹ā鍨嬨€佹暟鎹姞杞姐€侀獙璇佷笌宸ュ叿鍑芥暟銆?- `code_v4/dataloaders/`
  - 鏁版嵁闆嗗垏鍒嗐€佸急鏍囨敞璇诲彇銆佹暟鎹寮恒€佷笉鍚屾暟鎹泦棰勫鐞嗚剼鏈€?- `code_v4/networks/`
  - 鍒嗗壊缃戠粶涓庢ā鍨嬪伐鍘傦紝涓绘ā鍨嬪湪杩欓噷瀹氫箟銆?- `code_v4/utils/`
  - 鎹熷け鍑芥暟銆佹寚鏍囥€丆RF/TreeEnergyLoss 绛夎緟鍔╂ā鍧椼€?- `image/`
  - 璁烘枃/椤圭洰灞曠ず鍥剧墖锛屼笉鍙備笌璁粌涓婚€昏緫銆?- `fedlppa.yaml`
  - 鐜渚濊禆閰嶇疆銆?- `tree_filter-0.1-*.whl`
  - TreeEnergyLoss 鐩稿叧鑷畾涔変緷璧栧寘銆?
## 3. 鏍稿績妯″瀷鍦ㄥ摢閲?
褰撳墠椤圭洰涓荤嚎鐨勬牳蹇冩ā鍨嬫槸锛?
- `code_v4/networks/unet.py`
  - `class UNet_UniV5`

妯″瀷鏋勯€犲叆鍙ｅ湪锛?
- `code_v4/networks/net_factory.py`
  - `net_factory(args, net_type=..., in_chns=..., class_num=...)`

褰撳墠 FedLPPA 涓昏缁冭剼鏈粯璁ら€氳繃锛?
- `--model unet_univ5`

鎶?`UNet_UniV5` 瀹炰緥鍖栧嚭鏉ャ€?
`UNet_UniV5` 鐨勬牳蹇冪壒寰侊細

- 缂栫爜鍣ㄦ彁鍙栫壒寰侊紱
- 浣跨敤 `distribution_prompts` 琛ㄧず瀹㈡埛绔釜鎬?prompt锛?- 浣跨敤 `uni_prompt` 琛ㄧず鍏变韩 prompt锛?- 鍙€?`label_prompt` 琛ㄧず鐩戠潱绫诲瀷 prompt锛?- 鐢?attention 妯″潡铻嶅悎 prompt 鍜屾繁灞傜壒寰侊紱
- 杈撳嚭涓诲垎鍓插垎鏀?`output` 涓庤緟鍔╁垎鏀?`output_auxiliary`锛屼緵 FedLPPA 鐨勫弻鍒嗘敮浼爣绛惧涔犱娇鐢ㄣ€?
## 4. 璁粌鍏ュ彛鍦ㄥ摢閲?
涓昏缁冨叆鍙ｏ細

- `code_v4/flower_pCE_2D_v4_FedLPPA.py`
  - `main()`

鍚姩鏂瑰紡鐢卞弬鏁?`--role server/client` 鍐冲畾锛?
- `role=server`
  - 鍒涘缓 `strategy`銆乣MyServer`锛岀劧鍚?`fl.server.start_server(...)`
- `role=client`
  - 鍒涘缓 `MyClient`锛岀劧鍚?`fl.client.start_client(...)`

鍛戒护鏍蜂緥鍏ュ彛锛?
- `code_v4/train.sh`

鍘嗗彶/鍩虹嚎鍏ュ彛锛?
- `code_v4/flower_pCE_2D.py`
  - 杈冩棭鐗堟湰鍩虹嚎鍏ュ彛銆?- `code_v4/flower_pCE_2D_v4.py`
  - v4 鐗堣缁冨叆鍙ｃ€?
## 5. 鑱旈偊瀛︿範閫昏緫鍦ㄥ摢浜涙枃浠?
鑱旈偊瀛︿範涓婚€昏緫涓昏鍦ㄤ笅闈㈣繖浜涙枃浠讹細

- `code_v4/flower_common_v4.py`
  - 褰撳墠涓荤嚎鑱旈偊妗嗘灦鏂囦欢锛屽畾涔夊鎴风鍩虹被銆佹湇鍔＄寰幆銆佸弬鏁拌仛鍚堛€佷釜鎬у寲鍙傛暟瑁呰浇銆佺瓥鐣ラ€夋嫨銆?- `code_v4/flower_pCE_2D_v4_FedLPPA.py`
  - 褰撳墠 FedLPPA 璁粌鑴氭湰锛屽畾涔?`MyClient._train()` 涓殑鏈湴璁粌鎹熷け涓庝笂浼犳寚鏍囥€?- `code_v4/networks/unet.py`
  - 妯″瀷鍐呴儴 prompt/鍙屽垎鏀?attention 缁撴瀯锛屽喅瀹氬彲琚仈閭︿釜鎬у寲鐨勫弬鏁板舰鎬併€?- `code_v4/networks/net_factory.py`
  - 鏍规嵁鑱旈偊绛栫暐閫夋嫨鍏蜂綋缃戠粶銆?- `code_v4/flower_common.py`
  - 鏃х増鑱旈偊鍏叡閫昏緫鏂囦欢锛屼繚鐣欏巻鍙茬瓥鐣ュ疄鐜般€?- `code_v4/flower_common_v4_addprostate.py`
  - 闈㈠悜 prostate 鍦烘櫙鐨勫彉浣撹仈閭﹀叕鍏遍€昏緫銆?
鍏朵腑鏈€鍏抽敭鐨勮仈閭︾被/鍑芥暟鏄細

- `BaseClient`
  - Flower 瀹㈡埛绔皝瑁咃紝璐熻矗 `get_parameters / fit / evaluate`銆?- `MyServer`
  - 鑷畾涔夋湇鍔″櫒璁粌寰幆锛岃礋璐?round 绾?fit/eval銆佽褰曟棩蹇椼€侀噸缁勫弬鏁般€?- `get_strategy`
  - 鏍规嵁鍚嶅瓧鍒涘缓 `FedAvg/FedProx/FedAP/FedAN/FedUniV2...`銆?- `MyModel.get_weights / set_weights`
  - 瀹氫箟涓嶅悓绛栫暐涓嬪弬鏁板浣曚笂浼犲拰涓嬪彂銆?- `FedAP/FedAN/FedUniV2(...)`
  - 鍚勮仈閭︾瓥鐣ョ殑鑱氬悎鏂瑰紡銆?
## 6. 鏁版嵁娴侊細浠庤緭鍏ュ埌杈撳嚭鐨?pipeline

褰撳墠涓荤嚎 pipeline 濡備笅锛?
1. 杈撳叆鏁版嵁缁勭粐
   - 鏍圭洰褰曠敱 `--root_path` 鎸囧悜锛屽唴閮ㄦ寜 `Domain1/Domain2/...`銆乣train/test`銆乣.h5` 鏂囦欢缁勭粐銆?
2. 瀹㈡埛绔暟鎹垝鍒?   - `code_v4/dataloaders/dataset.py` 鐨?`BaseDataSets` 鏍规嵁 `client=client1/client2/...` 閫夋嫨瀵瑰簲鍩熺殑鏁版嵁銆?
3. 璇诲彇鏍锋湰
   - 姣忎釜 `.h5` 璇诲彇 `image`锛?   - 璁粌鏃舵寜 `sup_type` 璇诲彇瀵瑰簲鐩戠潱锛屼緥濡?`mask/scribble/keypoint/block/box`锛?   - 楠岃瘉鏃剁粺涓€璇诲彇 `mask` 浣滀负鐪熷€笺€?
4. 寮辨爣娉ㄥ鐞嗕笌澧炲己
   - `pseudo_label_generator_acdc(...)` 鍙妸绋€鐤忕瀛愯浆鎴愪吉鏍囩锛?   - `RandomGenerator` 鍋氶殢鏈虹炕杞€侀殢鏈烘棆杞紝骞惰浆鎴?tensor銆?
5. DataLoader 缁?batch
   - `flower_pCE_2D_v4_FedLPPA.py` 涓瀯閫?`trainloader/valloader`銆?
6. 妯″瀷鏋勫缓
   - `net_factory(...)` 鏍规嵁鍙傛暟鍒涘缓 `UNet_UniV5` 鎴栧叾浠?UNet 鍙樹綋銆?
7. 瀹㈡埛绔湰鍦拌缁?   - `MyClient._train()` 涓墠鍚戜紶鎾緱鍒帮細
     - 涓诲垎鍓茶緭鍑?`outputs`
     - 杈呭姪杈撳嚭 `outputs_auxiliary`
     - prompt 鐗瑰緛 `prompts/distribution_prompts/uni_prompts`
   - 璁＄畻鎹熷け锛?     - 涓昏緟鍔╁垎鏀?CE 鍒嗗壊鎹熷け
     - `FedUniV2/FedUniV2.1` 鐨?prompt 涓€鑷存€ф崯澶?`loss_uni`
     - 鍙屽垎鏀吉鏍囩绾︽潫 `loss_pls`
     - gated CRF 姝ｅ垯椤?     - 鍏朵粬绛栫暐瀵瑰簲棰濆鎹熷け

8. 涓婁紶鑱旈偊鍙傛暟
   - `MyModel.get_weights()` 瀵煎嚭鏈湴妯″瀷鍙傛暟锛?   - 瀹㈡埛绔悓鏃朵笂浼犺缁冩寚鏍囥€佸彲瑙嗗寲鍥惧儚銆乸rompt 绛変俊鎭€?
9. 鏈嶅姟绔仛鍚?   - `MyServer.fit()` 椹卞姩鑱旈偊杞锛?   - `strategy.aggregate_fit(...)` 鍋氱瓥鐣ョ浉鍏宠仛鍚堬紱
   - 瀵?`FedUniV2/FedUniV2.1`锛屾湇鍔″櫒浼氶澶栨嫾鎺ワ細
     - 姣忎釜瀹㈡埛绔潈閲?     - 褰撳墠 server 鏉冮噸
     - 楠岃瘉鎬ц兘鏁扮粍
     - 骞冲潎 prompt 鏁扮粍

10. 涓€у寲鍙傛暟鍥炵亴
   - `MyModel.set_weights()` 鎸夌瓥鐣ユ媶瑙ｆ湇鍔″櫒涓嬪彂鍙傛暟锛?   - `FedUniV2/FedUniV2.1` 涓細缁撳悎锛?     - 鍏ㄥ眬 server 鍙傛暟
     - 瀹㈡埛绔嚜韬弬鏁?     - 鑱氬悎鍚庣殑 prompt
     - `dual_init` 瑙勫垯
   - 涓烘瘡涓鎴风鐢熸垚甯︿釜鎬у寲鍒濆鍖栫殑鏈湴妯″瀷銆?
11. 楠岃瘉涓庤緭鍑?   - `val_2D.py` 鐨?`test_single_volume(...)` 鐢熸垚棰勬祴锛?   - 璁＄畻 `dice/hd95/recall/precision/jc/specificity/ravd`锛?   - 淇濆瓨鏈€浼?checkpoint 鍒?`snapshot_path`锛?   - TensorBoard 璁板綍璁粌鍥惧儚銆乼sne銆佸悇瀹㈡埛绔寚鏍囥€?
# 鍏抽敭鏂囦欢鍒楄〃

## 褰撳墠涓荤嚎鍏抽敭鏂囦欢

- `code_v4/flower_pCE_2D_v4_FedLPPA.py`
  - 褰撳墠椤圭洰涓昏缁冨叆鍙ｏ紝璐熻矗鍙傛暟瑙ｆ瀽銆佹暟鎹姞杞姐€佹ā鍨嬪垱寤恒€丗lower server/client 鍚姩锛屼互鍙婃湰鍦拌缁冩崯澶卞畾涔夈€?- `code_v4/flower_common_v4.py`
  - 褰撳墠涓荤嚎鑱旈偊鍏叡閫昏緫锛岃礋璐ｅ鎴风閫氫俊銆佹湇鍔＄寰幆銆佺瓥鐣ュ垱寤恒€佽仛鍚堜笌涓€у寲鍙傛暟涓嬪彂銆?- `code_v4/networks/net_factory.py`
  - 妯″瀷宸ュ巶锛屾牴鎹?`args.model` 瀹炰緥鍖?`UNet_UniV5` 绛夌綉缁溿€?- `code_v4/networks/unet.py`
  - 涓昏缃戠粶瀹氫箟鏂囦欢锛屽寘鍚?`UNet_UniV5` 鍙婂涓釜鎬у寲 UNet 鍙樹綋銆?- `code_v4/dataloaders/dataset.py`
  - 涓绘暟鎹泦绫伙紝璐熻矗鎸夊鎴风鍒掑垎鍩熸暟鎹€佽鍙栧急鏍囨敞銆佹暟鎹寮哄拰 tensor 鍖栥€?- `code_v4/val_2D.py`
  - 楠岃瘉涓庢祴璇曢€昏緫锛岃礋璐ｆ妸妯″瀷杈撳嚭杞负鍒嗗壊缁撴灉骞惰绠楁寚鏍囥€?- `code_v4/train.sh`
  - 鎻愪緵 FedAvg 涓?FedLPPA 鐨勫疄闄呭惎鍔ㄥ懡浠ゃ€?
## 閲嶈杈呭姪鏂囦欢

- `code_v4/flower_pCE_2D.py`
  - 鏃╂湡/鍩虹嚎鑱旈偊璁粌鍏ュ彛銆?- `code_v4/flower_pCE_2D_v4.py`
  - v4 鐗堣缁冨叆鍙ｏ紝鍜?FedLPPA 涓绘枃浠跺悓鏃忋€?- `code_v4/flower_common.py`
  - 鏃х増鑱旈偊鍏叡閫昏緫锛屼繚鐣欎簡鐩镐技鐨勮仈閭﹀疄鐜版鏋躲€?- `code_v4/flower_common_v4_addprostate.py`
  - 閽堝 prostate 鏁版嵁鍦烘櫙鎵╁睍鐨勮仈閭﹂€昏緫鐗堟湰銆?- `code_v4/test_client4onemod_FL_Personalize.py`
  - 涓€у寲妯″瀷娴嬭瘯/瀵煎嚭鑴氭湰銆?- `code_v4/utils/losses.py`
  - 甯哥敤鍒嗗壊鎹熷け锛屽 Dice 绫绘崯澶便€?- `code_v4/utils/gate_crf_loss.py`
  - FedLPPA 璁粌閲屼娇鐢ㄧ殑 gated CRF 姝ｅ垯椤广€?- `code_v4/utils/TreeEnergyLoss/...`
  - TreeEnergyLoss 鐩稿叧瀹炵幇涓?CUDA 鎵╁睍銆?
# 妯″潡涓€鍙ヨ瘽璇存槑

## 椤跺眰妯″潡

- `code_v4/`锛氶」鐩富浠ｇ爜鐩綍锛屽寘鍚缁冦€佽仈閭﹀涔犮€佹ā鍨嬨€佹暟鎹拰楠岃瘉閫昏緫銆?- `image/`锛氳鏂囩粨鏋滀笌妗嗘灦绀烘剰鍥剧洰褰曪紝涓嶅弬涓庤缁冩墽琛屻€?- `fedlppa.yaml`锛歝onda 鐜渚濊禆閰嶇疆鏂囦欢銆?- `tree_filter-0.1-cp39-cp39-linux_x86_64.whl`锛歍reeEnergyLoss 鎵€闇€鐨勮嚜瀹氫箟浜岃繘鍒朵緷璧栧寘銆?
## code_v4 涓昏剼鏈ā鍧?
- `code_v4/flower_pCE_2D_v4_FedLPPA.py`锛欶edLPPA 褰撳墠涓诲叆鍙ｏ紝闆嗘垚 server/client 鍚姩涓庢湰鍦拌缁冭繃绋嬨€?- `code_v4/flower_pCE_2D_v4.py`锛歷4 鐗堝疄楠屽叆鍙ｏ紝鎵胯浇 FedLPPA 涔嬪墠鎴栧苟琛岀殑璁粌涓荤嚎銆?- `code_v4/flower_pCE_2D.py`锛氭洿鏃╀竴鐗堣仈閭﹁缁冨叆鍙ｏ紝涓昏鐢ㄤ簬鍩虹嚎鏂规硶銆?- `code_v4/flower_common_v4.py`锛氬綋鍓嶇増鏈仈閭﹀叕鍏辨鏋朵笌绛栫暐瀹炵幇涓績銆?- `code_v4/flower_common_v4_addprostate.py`锛氶€傞厤 prostate 鏁版嵁鐨勮仈閭﹀叕鍏遍€昏緫鍙樹綋銆?- `code_v4/flower_common.py`锛氭棫鐗堣仈閭﹀叕鍏辨鏋舵枃浠躲€?- `code_v4/val_2D.py`锛?D 鍒嗗壊楠岃瘉鍜屾寚鏍囪绠楁ā鍧椼€?- `code_v4/train.sh`锛氳缁冨懡浠よ剼鏈紝灞曠ず server 涓庡 client 鐨勫惎鍔ㄦ柟寮忋€?- `code_v4/test.sh`锛氭祴璇曞懡浠よ剼鏈€?- `code_v4/test_client4onemod_FL_Personalize.py`锛氫釜鎬у寲妯″瀷娴嬭瘯涓庣粨鏋滃鍑鸿剼鏈€?- `code_v4/flower_command.sh`锛氳仈閭﹁缁冨懡浠よ緟鍔╄剼鏈€?
## dataloaders 妯″潡

- `code_v4/dataloaders/dataset.py`锛氫富鏁版嵁闆嗗畾涔夛紝璐熻矗鑱旈偊瀹㈡埛绔垝鍒嗐€佸急鏍囨敞璇诲彇涓庡寮恒€?- `code_v4/dataloaders/utils.py`锛氭暟鎹姞杞借緟鍔╁嚱鏁般€?- `code_v4/dataloaders/Unet_pCE.py`锛氬崟鏈?U-Net 璁粌鑴氭湰鎴栨棭鏈熸暟鎹缁冨叆鍙ｃ€?- `code_v4/dataloaders/dataset_fully.py`锛氬叏鐩戠潱鏁版嵁闆嗗畾涔夈€?- `code_v4/dataloaders/dataset_semi.py`锛氬崐鐩戠潱鏁版嵁闆嗗畾涔夈€?- `code_v4/dataloaders/dataset_ft.py`锛氬井璋冨満鏅暟鎹泦瀹氫箟銆?- `code_v4/dataloaders/dataset_rw.py`锛氶殢鏈烘父璧颁吉鏍囩鐩稿叧鏁版嵁闆嗛€昏緫銆?- `code_v4/dataloaders/dataset_s2l.py`锛氱█鐤忔爣娉ㄥ埌鏍囩杞崲鐩稿叧鏁版嵁闆嗛€昏緫銆?- `code_v4/dataloaders/dataset_CL.py`锛氬姣?璇剧▼寮忚缁冪浉鍏虫暟鎹泦閫昏緫銆?- `code_v4/dataloaders/acdc_pseudo_label_random_walker.py`锛氬熀浜?random walker 鐢熸垚浼爣绛剧殑鑴氭湰銆?- `code_v4/dataloaders/icctw_data_processing.py`锛欼CCTW 鏁版嵁棰勫鐞嗚剼鏈€?- `code_v4/dataloaders/odocfaz_data_processing.py`锛歄DOC/FAZ 鏁版嵁棰勫鐞嗚剼鏈€?
## networks 妯″潡

- `code_v4/networks/net_factory.py`锛?D 缃戠粶宸ュ巶锛岀粺涓€鍒涘缓涓嶅悓鍒嗗壊妯″瀷銆?- `code_v4/networks/net_factory_3d.py`锛?D 缃戠粶宸ュ巶銆?- `code_v4/networks/unet.py`锛氶」鐩渶鏍稿績鐨?2D UNet 鍙?prompt 涓€у寲鍙樹綋瀹氫箟鏂囦欢銆?- `code_v4/networks/unet_3D.py`锛?D UNet 瀹炵幇銆?- `code_v4/networks/vnet.py`锛歏Net 瀹炵幇銆?- `code_v4/networks/VoxResNet.py`锛歏oxResNet 瀹炵幇銆?- `code_v4/networks/attention_unet.py`锛欰ttention U-Net 瀹炵幇銆?- `code_v4/networks/attention.py`锛氭敞鎰忓姏妯″潡瀹炵幇銆?- `code_v4/networks/grid_attention_layer.py`锛氱綉鏍兼敞鎰忓姏灞傚疄鐜般€?- `code_v4/networks/pnet.py`锛歅Net2D 鍒嗗壊缃戠粶瀹炵幇銆?- `code_v4/networks/efficientunet.py`锛欵fficientNet 缂栫爜鍣ㄧ増 U-Net 瀹炵幇銆?- `code_v4/networks/efficient_encoder.py`锛欵fficientNet 缂栫爜鍣ㄥ畾涔夈€?- `code_v4/networks/encoder_tool.py`锛氱紪鐮佸櫒宸ュ叿涓庡皝瑁呬唬鐮併€?- `code_v4/networks/networks_other.py`锛氬叾浠栫敓鎴愬櫒/鍒ゅ埆鍣?UNet 鍙樹綋闆嗗悎銆?- `code_v4/networks/discriminator.py`锛氬垽鍒櫒缃戠粶瀹氫箟銆?- `code_v4/networks/utils.py`锛氱綉缁滅浉鍏冲伐鍏峰嚱鏁般€?
## utils 妯″潡

- `code_v4/utils/losses.py`锛氶」鐩父鐢ㄦ崯澶卞嚱鏁板畾涔夈€?- `code_v4/utils/metrics.py`锛氳缁冩垨璇勪及闃舵鐨勬寚鏍囧嚱鏁般€?- `code_v4/utils/ramps.py`锛氳缁冭繃绋嬩腑鐨?ramp-up/ramp-down 璋冨害鍑芥暟銆?- `code_v4/utils/util.py`锛氶€氱敤宸ュ叿鍑芥暟銆?- `code_v4/utils/custom_transforms.py`锛氬浘鍍忓彉鎹㈠伐鍏枫€?- `code_v4/utils/gate_crf_loss.py`锛欸ated CRF 姝ｅ垯鎹熷け瀹炵幇銆?- `code_v4/utils/DenseCRFLoss.py`锛欴ense CRF 鎹熷け瀹炵幇銆?- `code_v4/utils/AGEnergyLoss.py`锛氳兘閲忔崯澶辩浉鍏冲疄鐜般€?- `code_v4/utils/TreeEnergyLoss/`锛歍ree Energy Loss 鐨勫畬鏁村疄鐜般€侀厤缃€佽剼鏈拰 CUDA 鏍稿績銆?- `code_v4/utils/pytorch/`锛氱涓夋柟鎴栫嫭绔嬪疄楠屾€ц川鐨?PyTorch 瀛愬伐绋嬮泦鍚堛€?
# 鏈€鍏抽敭鐨勭粨璁?
- 浠诲姟鏈川锛氳仈閭﹀急鐩戠潱鍖诲鍥惧儚鍒嗗壊锛屾牳蹇冮棶棰樻槸鈥滆法瀹㈡埛绔紓鏋勫急鏍囨敞 + 鍩熷樊寮?+ 涓€у寲鑱氬悎鈥濄€?- 褰撳墠涓绘ā鍨嬶細`code_v4/networks/unet.py` 涓殑 `UNet_UniV5`銆?- 褰撳墠璁粌鍏ュ彛锛歚code_v4/flower_pCE_2D_v4_FedLPPA.py` 鐨?`main()`銆?- 褰撳墠鑱旈偊涓婚€昏緫锛歚code_v4/flower_common_v4.py`銆?- 褰撳墠涓绘暟鎹?pipeline锛歚BaseDataSets -> RandomGenerator -> DataLoader -> net_factory/UNet_UniV5 -> MyClient._train -> Flower 鑱氬悎 -> MyModel.set_weights 涓€у寲涓嬪彂 -> val_2D 璇勪及`銆?
