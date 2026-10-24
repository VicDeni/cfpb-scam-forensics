# Labeling guide: payment-scam complaints

Generated from `complaint_kit/scam.py`. The LLM gets exactly these definitions, so you and the model are judged by the same rules.

**Label every row before you look at any model output.** Seeing the model's answer first biases you toward agreeing with it, which makes its accuracy look better than it is.

The easiest way to label is `python scripts/label_scams.py`, which shows one complaint at a time, lets you pick each value by number, applies the rules below for you, and saves as you go. If you edit the sheet by hand instead, type values exactly as written below. Blank cells are skipped when scoring. Use `notes` for anything ambiguous: those cases are good material for the error analysis.

## Rules

- If fraud_type is not_fraud, every other field is not_applicable and the amount is null.
- If fraud_type is unauthorized_transaction, scam_type and contact_channel are not_applicable.
- Judge only from what the consumer wrote. Don't assume details that aren't there; use unclear or not_mentioned instead.
- Text marked [REDACTED] was removed for privacy; [PHONE], [EMAIL] and similar are masked values.

## `fraud_type`

What kind of problem is this?

| Value | Meaning |
|---|---|
| `authorized_scam` | The consumer was tricked into sending the money or approving the payment themselves, e.g. someone posing as their bank told them to move money to a 'safe account', or they paid a fake seller. |
| `unauthorized_transaction` | Money left the account without the consumer's action or approval: stolen card or phone, hacked account, a payment they did not make. |
| `not_fraud` | No fraud or scam is described, e.g. a fee dispute, a delayed or stuck transfer, or a disagreement with a real merchant. |

## `scam_type`

If authorized_scam: what was the story the scammer used?

| Value | Meaning |
|---|---|
| `bank_impersonation` | Posed as the consumer's bank or its fraud department. |
| `government_impersonation` | Posed as a government agency (IRS, Social Security, police, courts). |
| `business_impersonation` | Posed as a well-known company or tech support (Amazon, Apple, Microsoft, a utility). |
| `romance` | An online relationship used to ask for money. |
| `investment_crypto` | A fake investment, trading or crypto opportunity. |
| `purchase_or_sale` | A fake seller or buyer: marketplace item, rental, tickets, pets. |
| `job_or_income` | A fake job, overpayment, or money-making scheme. |
| `family_or_friend` | Posed as a relative or friend in an emergency. |
| `other` | A scam that fits none of the above. |
| `unclear` | A scam, but not enough detail to tell which kind. |
| `not_applicable` | Not an authorized_scam. |

## `contact_channel`

If authorized_scam: how did the scammer first reach the consumer?

| Value | Meaning |
|---|---|
| `phone_call` | Phone call. |
| `text_message` | Text message. |
| `email` | Email. |
| `social_media` | Social media or messaging app. |
| `dating_site` | Dating app or site. |
| `online_marketplace` | Online marketplace or classifieds listing. |
| `website_or_ad` | A website, search result or online ad. |
| `other` | Some other way (in person, mail). |
| `unclear` | Not stated. |
| `not_applicable` | Not an authorized_scam. |

## `payment_method`

If fraud: how did the money leave?

| Value | Meaning |
|---|---|
| `zelle` | Zelle. |
| `other_p2p_app` | Another payment app: Cash App, Venmo, PayPal, Apple Cash. |
| `wire_transfer` | Bank wire transfer. |
| `crypto` | Cryptocurrency, including crypto ATMs. |
| `card` | Debit or credit card payment. |
| `bank_transfer` | ACH or other account-to-account transfer, or online bill pay. |
| `check` | Check or money order. |
| `gift_card` | Gift cards. |
| `cash_withdrawal` | Cash withdrawn and handed over or deposited. |
| `multiple` | More than one of these. |
| `unclear` | Not stated. |
| `not_applicable` | Not fraud. |

## `refund_status`

If fraud: what had the bank or payment company done about the loss when the consumer wrote the complaint?

| Value | Meaning |
|---|---|
| `refunded_full` | Refunded all of it. |
| `refunded_partial` | Refunded some of it. |
| `denied` | Refused to refund, or said the consumer is responsible. |
| `pending` | Still investigating, or the consumer is waiting. |
| `not_mentioned` | The complaint doesn't say. |
| `not_applicable` | Not fraud. |

## `amount_lost_usd`

Total US dollars the consumer says they lost or sent to the fraudster, as a number (e.g. 1250.00). Add up multiple payments. Leave empty (null) if no amount is stated or it isn't fraud.

In the sheet, type `none` when no amount is stated (a blank cell means you skipped it). Plain numbers or `$1,250.00` both work.
