# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ISO 4217 currency normalization for credit-report projections.

The code registry is a static snapshot of ISO 4217 List One published by the
SIX maintenance agency on 2026-01-01.  Chinese aliases cover the currency
labels used by domestic credit reports.  Unknown explicit account currencies
are deliberately preserved by the caller instead of being mislabeled as CNY.
"""

from __future__ import annotations

import re

ISO_4217_LIST_ONE_PUBLISHED = "2026-01-01"

ISO_4217_CURRENT_CODES = frozenset(
    """
    AED AFN ALL AMD AOA ARS AUD AWG AZN BAM BBD BDT BHD BIF BMD BND
    BOB BOV BRL BSD BTN BWP BYN BZD CAD CDF CHE CHF CHW CLF CLP CNY
    COP COU CRC CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP
    GBP GEL GHS GIP GMD GNF GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD
    IRR ISK JMD JOD JPY KES KGS KHR KMF KPW KRW KWD KYD KZT LAK LBP
    LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR MVR MWK MXN
    MXV MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN
    PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD
    SSP STN SVC SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX
    USD USN UYI UYU UYW UZS VED VES VND VUV WST XAD XAF XAG XAU XBA
    XBB XBC XBD XCD XCG XDR XOF XPD XPF XPT XSU XTS XUA XXX YER ZAR
    ZMW ZWG
    """.split()
)

# ISO codes are always accepted directly. These aliases translate the names
# that can be printed in Chinese credit-report account and enterprise tables.
_ALIASES_BY_CODE: dict[str, tuple[str, ...]] = {
    "AED": ("阿联酋迪拉姆", "阿拉伯联合酋长国迪拉姆"),
    "AFN": ("阿富汗尼",),
    "ALL": ("阿尔巴尼亚列克", "列克"),
    "AMD": ("亚美尼亚德拉姆",),
    "AOA": ("安哥拉宽扎", "宽扎"),
    "ARS": ("阿根廷比索",),
    "AUD": ("澳大利亚元", "澳元"),
    "AWG": ("阿鲁巴弗罗林",),
    "AZN": ("阿塞拜疆马纳特",),
    "BAM": ("波黑可兑换马克", "波斯尼亚和黑塞哥维那可兑换马克"),
    "BBD": ("巴巴多斯元",),
    "BDT": ("孟加拉塔卡", "孟加拉国塔卡"),
    "BHD": ("巴林第纳尔",),
    "BIF": ("布隆迪法郎",),
    "BMD": ("百慕大元",),
    "BND": ("文莱元",),
    "BOB": ("玻利维亚诺", "玻利维亚币"),
    "BOV": ("玻利维亚MVDOL", "玻利维亚基金单位"),
    "BRL": ("巴西雷亚尔",),
    "BSD": ("巴哈马元",),
    "BTN": ("不丹努尔特鲁姆", "努尔特鲁姆"),
    "BWP": ("博茨瓦纳普拉", "普拉"),
    "BYN": ("白俄罗斯卢布",),
    "BZD": ("伯利兹元",),
    "CAD": ("加拿大元", "加元"),
    "CDF": ("刚果法郎",),
    "CHE": ("WIR欧元", "互助欧元"),
    "CHF": ("瑞士法郎",),
    "CHW": ("WIR法郎", "互助法郎"),
    "CLF": ("智利发展单位", "智利记账单位"),
    "CLP": ("智利比索",),
    "CNY": ("人民币元", "人民币", "中国元"),
    "COP": ("哥伦比亚比索",),
    "COU": ("哥伦比亚真实价值单位", "哥伦比亚价值单位"),
    "CRC": ("哥斯达黎加科朗",),
    "CUP": ("古巴比索",),
    "CVE": ("佛得角埃斯库多",),
    "CZK": ("捷克克朗",),
    "DJF": ("吉布提法郎",),
    "DKK": ("丹麦克朗",),
    "DOP": ("多米尼加比索",),
    "DZD": ("阿尔及利亚第纳尔",),
    "EGP": ("埃及镑",),
    "ERN": ("厄立特里亚纳克法", "纳克法"),
    "ETB": ("埃塞俄比亚比尔",),
    "EUR": ("欧元",),
    "FJD": ("斐济元",),
    "FKP": ("福克兰群岛镑",),
    "GBP": ("英镑",),
    "GEL": ("格鲁吉亚拉里", "拉里"),
    "GHS": ("加纳塞地",),
    "GIP": ("直布罗陀镑",),
    "GMD": ("冈比亚达拉西", "达拉西"),
    "GNF": ("几内亚法郎",),
    "GTQ": ("危地马拉格查尔", "格查尔"),
    "GYD": ("圭亚那元",),
    "HKD": ("香港元", "港币", "港元"),
    "HNL": ("洪都拉斯伦皮拉", "伦皮拉"),
    "HTG": ("海地古德", "古德"),
    "HUF": ("匈牙利福林", "福林"),
    "IDR": ("印度尼西亚卢比", "印尼盾", "印度尼西亚盾"),
    "ILS": ("以色列新谢克尔", "新谢克尔", "以色列谢克尔"),
    "INR": ("印度卢比",),
    "IQD": ("伊拉克第纳尔",),
    "IRR": ("伊朗里亚尔",),
    "ISK": ("冰岛克朗",),
    "JMD": ("牙买加元",),
    "JOD": ("约旦第纳尔",),
    "JPY": ("日元", "日圆"),
    "KES": ("肯尼亚先令",),
    "KGS": ("吉尔吉斯斯坦索姆", "吉尔吉斯索姆"),
    "KHR": ("柬埔寨瑞尔",),
    "KMF": ("科摩罗法郎",),
    "KPW": ("朝鲜元", "朝鲜圆"),
    "KRW": ("韩元", "韩国元", "韩国圆"),
    "KWD": ("科威特第纳尔",),
    "KYD": ("开曼群岛元", "开曼元"),
    "KZT": ("哈萨克斯坦坚戈", "哈萨克坚戈"),
    "LAK": ("老挝基普",),
    "LBP": ("黎巴嫩镑",),
    "LKR": ("斯里兰卡卢比",),
    "LRD": ("利比里亚元",),
    "LSL": ("莱索托洛蒂", "洛蒂"),
    "LYD": ("利比亚第纳尔",),
    "MAD": ("摩洛哥迪拉姆",),
    "MDL": ("摩尔多瓦列伊", "摩尔多瓦列伊"),
    "MGA": ("马达加斯加阿里亚里", "阿里亚里"),
    "MKD": ("北马其顿代纳尔", "马其顿代纳尔"),
    "MMK": ("缅甸元", "缅元", "缅甸缅元"),
    "MNT": ("蒙古图格里克", "图格里克"),
    "MOP": ("澳门元", "澳门币", "澳币"),
    "MRU": ("毛里塔尼亚乌吉亚", "乌吉亚"),
    "MUR": ("毛里求斯卢比",),
    "MVR": ("马尔代夫拉菲亚", "拉菲亚"),
    "MWK": ("马拉维克瓦查",),
    "MXN": ("墨西哥比索",),
    "MXV": ("墨西哥投资单位",),
    "MYR": ("马来西亚林吉特", "马来西亚元", "马币"),
    "MZN": ("莫桑比克梅蒂卡尔", "梅蒂卡尔"),
    "NAD": ("纳米比亚元",),
    "NGN": ("尼日利亚奈拉", "奈拉"),
    "NIO": ("尼加拉瓜科多巴", "科多巴"),
    "NOK": ("挪威克朗",),
    "NPR": ("尼泊尔卢比",),
    "NZD": ("新西兰元", "纽元"),
    "OMR": ("阿曼里亚尔",),
    "PAB": ("巴拿马巴波亚", "巴波亚"),
    "PEN": ("秘鲁索尔", "新索尔"),
    "PGK": ("巴布亚新几内亚基那", "巴新基那", "基那"),
    "PHP": ("菲律宾比索",),
    "PKR": ("巴基斯坦卢比",),
    "PLN": ("波兰兹罗提", "兹罗提"),
    "PYG": ("巴拉圭瓜拉尼", "瓜拉尼"),
    "QAR": ("卡塔尔里亚尔",),
    "RON": ("罗马尼亚列伊",),
    "RSD": ("塞尔维亚第纳尔",),
    "RUB": ("俄罗斯卢布", "俄卢布", "卢布"),
    "RWF": ("卢旺达法郎",),
    "SAR": ("沙特里亚尔", "沙特阿拉伯里亚尔"),
    "SBD": ("所罗门群岛元",),
    "SCR": ("塞舌尔卢比",),
    "SDG": ("苏丹镑",),
    "SEK": ("瑞典克朗",),
    "SGD": ("新加坡元", "新元"),
    "SHP": ("圣赫勒拿镑",),
    "SLE": ("塞拉利昂利昂", "利昂"),
    "SOS": ("索马里先令",),
    "SRD": ("苏里南元",),
    "SSP": ("南苏丹镑",),
    "STN": ("圣多美和普林西比多布拉", "圣多美多布拉", "多布拉"),
    "SVC": ("萨尔瓦多科朗",),
    "SYP": ("叙利亚镑",),
    "SZL": ("埃斯瓦蒂尼里兰吉尼", "斯威士兰里兰吉尼", "里兰吉尼"),
    "THB": ("泰铢",),
    "TJS": ("塔吉克斯坦索莫尼", "索莫尼"),
    "TMT": ("土库曼斯坦马纳特", "土库曼马纳特"),
    "TND": ("突尼斯第纳尔",),
    "TOP": ("汤加潘加", "潘加"),
    "TRY": ("土耳其里拉", "新土耳其里拉"),
    "TTD": ("特立尼达和多巴哥元", "特多元"),
    "TWD": ("新台币", "新臺幣", "台湾元", "臺灣元"),
    "TZS": ("坦桑尼亚先令",),
    "UAH": ("乌克兰格里夫纳", "乌克兰赫里夫纳", "格里夫纳", "赫里夫纳"),
    "UGX": ("乌干达先令",),
    "USD": ("美元", "美金"),
    "USN": ("美元次日", "美元次日资金"),
    "UYI": ("乌拉圭指数单位",),
    "UYU": ("乌拉圭比索",),
    "UYW": ("乌拉圭名义工资指数单位",),
    "UZS": ("乌兹别克斯坦苏姆", "乌兹别克苏姆"),
    "VED": ("委内瑞拉数字玻利瓦尔",),
    "VES": ("委内瑞拉主权玻利瓦尔", "委内瑞拉玻利瓦尔"),
    "VND": ("越南盾", "越南东"),
    "VUV": ("瓦努阿图瓦图", "瓦图"),
    "WST": ("萨摩亚塔拉", "塔拉"),
    "XAD": ("阿拉伯记账第纳尔",),
    "XAF": ("中非法郎", "中非金融合作法郎"),
    "XAG": ("白银", "银"),
    "XAU": ("黄金", "金"),
    "XBA": ("欧洲复合单位",),
    "XBB": ("欧洲货币单位",),
    "XBC": ("欧洲记账单位9",),
    "XBD": ("欧洲记账单位17",),
    "XCD": ("东加勒比元",),
    "XCG": ("加勒比盾", "加勒比荷兰盾", "加勒比吉尔德"),
    "XDR": ("特别提款权",),
    "XOF": ("西非法郎", "西非金融共同体法郎"),
    "XPD": ("钯", "钯金"),
    "XPF": ("太平洋法郎", "法属太平洋法郎"),
    "XPT": ("铂", "铂金", "白金"),
    "XSU": ("苏克雷", "SUCRE"),
    "XTS": ("测试货币代码",),
    "XUA": ("非洲开发银行记账单位",),
    "XXX": ("无货币", "无币种"),
    "YER": ("也门里亚尔",),
    "ZAR": ("南非兰特", "兰特"),
    "ZMW": ("赞比亚克瓦查",),
    "ZWG": ("津巴布韦金", "津巴布韦黄金"),
}

CURRENCY_CODE_BY_ALIAS = {
    re.sub(r"\s+", "", alias).upper(): code
    for code, aliases in _ALIASES_BY_CODE.items()
    for alias in aliases
}
_SORTED_ALIASES = tuple(sorted(CURRENCY_CODE_BY_ALIAS, key=len, reverse=True))


def normalize_currency_code(value: str) -> str:
    """Return an ISO 4217 code for a recognized code or Chinese currency name."""
    compact = re.sub(r"\s+", "", str(value or "")).strip("()（）[]【】,，;；:：")
    if not compact or compact == "--":
        return ""

    upper = compact.upper()
    for token in re.findall(r"(?<![A-Z])([A-Z]{3})(?![A-Z])", upper):
        if token in ISO_4217_CURRENT_CODES:
            return token

    for alias in _SORTED_ALIASES:
        if alias in upper:
            return CURRENCY_CODE_BY_ALIAS[alias]
    return ""


__all__ = [
    "CURRENCY_CODE_BY_ALIAS",
    "ISO_4217_CURRENT_CODES",
    "ISO_4217_LIST_ONE_PUBLISHED",
    "normalize_currency_code",
]
