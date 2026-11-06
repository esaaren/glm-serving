#!/usr/bin/env python3
"""
Generates realistic Legal Context benchmark datasets for vLLM throughput benchmarking
of zai-org/GLM-5.3-Flash (and zai-org/GLM-5.3-Flash-BF16).

Use case: Enterprise M&A Due Diligence, Commercial Contract Redlining & Litigation Discovery Review.
Provides:
1. Detailed Legal System Prompt (Senior M&A and Commercial Transactions Counsel role)
2. Realistic multi-section legal input documents (Stock Purchase Agreements, Master Services
   Agreements, IP Cross-Licenses, Credit Agreements, Data Processing Addenda, Deposition Exhibits)
   calibrated to target token lengths (e.g. 4K, 8K, 16K, 32K, 64K, 128K, 200K+ tokens).
3. Structured Legal Output Prompts / reference completions (Risk Matrix, Redline Clauses,
   Indemnification Exposure Calculation, and Executive Counsel Memorandum) for both
   `vllm bench serve --dataset-name custom` (JSONL) and direct async OpenAI chat completions.
"""

import argparse
import json
import os
import random
from pathlib import Path

LEGAL_SYSTEM_PROMPT = """You are Principal Legal Counsel and Head of M&A Due Diligence & Complex Commercial Litigation at an AmLaw 10 firm.
Your mandate is to conduct an exhaustive, statute- and clause-level legal analysis of the provided transaction documents, commercial contracts, regulatory filings, and evidentiary exhibits.

For every matter submitted, you must produce a comprehensive, publication-grade Legal Review & Redline Memorandum structured into the following five mandatory sections:
1. EXECUTIVE RISK SUMMARY & MATERIALITY MATRIX: Classify each identified legal, financial, or regulatory exposure by severity (CRITICAL / HIGH / MODERATE / LOW), citing exact Section, Article, and Clause numbers.
2. CHANGE-OF-CONTROL, ASSIGNMENT & TERMINATION TRIGGERS: Identify all anti-assignment provisions, deemed-assignment clauses upon merger or change of control (>20% voting equity shift), exclusivity covenants, and termination-for-convenience / cross-default triggers.
3. INDEMNIFICATION, LIABILITY CAPS & SURVIVAL ANALYSIS: Audit all indemnification baskets (tipping vs. true deductible), per-claim mini-baskets, general vs. fundamental representation caps, super-caps for IP/data-breach/tax claims, and survival periods under Delaware and New York law.
4. INTELLECTUAL PROPERTY, DATA PRIVACY & REGULATORY COMPLIANCE: Evaluate background vs. foreground IP ownership, derivative works, open-source contamination warranties, GDPR/CCPA/HIPAA data transfer mechanisms, and Hart-Scott-Rodino (HSR) / CFIUS antitrust and national security implications.
5. BINDING REDLINE AMENDMENTS & REMEDIATION LANGUAGE: Draft exact, legally enforceable replacement contract language (in formal blackline/redline format showing [DELETED] and <<INSERTED>> text) to cure every CRITICAL and HIGH risk exposure in favor of the Acquirer/Client."""

CLAUSE_TEMPLATES = [
    """ARTICLE {art_num}. REPRESENTATIONS, WARRANTIES AND COVENANTS OF THE TARGET COMPANY
Section {art_num}.01 Organization, Standing and Corporate Power.
The Company and each of its Subsidiaries (a) is a corporation or limited liability company duly organized, validly existing and in good standing under the Laws of the State of Delaware (or the jurisdiction of its organization as set forth in Section {art_num}.01 of the Disclosure Schedules), (b) has all requisite corporate or organizational power and authority to carry on its business as now being conducted and as proposed to be conducted following the Closing Date, and (c) is duly qualified or licensed to do business and is in good standing in each jurisdiction in which the nature of its business or the ownership, leasing or operation of its properties makes such qualification or licensing necessary, except where the failure to be so qualified or licensed would not, individually or in the aggregate, reasonably be expected to result in a Company Material Adverse Effect (defined herein as an adverse economic impact exceeding ${mae_thresh} million over any rolling four-quarter fiscal period).

Section {art_num}.02 Capitalization and Voting Indebtedness.
The authorized capital stock of the Company consists of {auth_common},000,000 shares of Common Stock, par value $0.0001 per share, and {auth_pref},000,000 shares of Preferred Stock, par value $0.0001 per share. As of the Measurement Time, (i) {issued_common},450,820 shares of Common Stock were issued and outstanding, (ii) {issued_pref},120,000 shares of Preferred Stock were issued and outstanding, (iii) {options_pool},890,000 shares of Common Stock were reserved for issuance pursuant to the Company Equity Incentive Plans, and (iv) warrants to purchase {warrants},500,000 shares of Common Stock were outstanding. Except as set forth in Schedule {art_num}.02(b), there are no bonds, debentures, notes or other indebtedness of the Company having the right to vote (or convertible into, or exchangeable for, securities having the right to vote) on any matters on which stockholders of the Company may vote ("Voting Debt").""",

    """ARTICLE {art_num}. INTELLECTUAL PROPERTY, AI TRAINING DATA & TECHNOLOGY LICENSING
Section {art_num}.01 Ownership of Company Intellectual Property.
Except as disclosed in Section {art_num}.01 of the Company Disclosure Letter, the Company and its Subsidiaries exclusively own all right, title, and interest in and to all Company Registered IP and all proprietary foundational models, weights, checkpoints, fine-tuning datasets, and inference pipelines ("Proprietary AI Assets"), free and clear of all Encumbrances. HOWEVER, pursuant to that certain Joint Development Agreement dated {jda_date} by and between the Company and {counterparty} ("Counterparty"), any improvements, derivative architectures, or fine-tuned adapters developed using Counterparty's telemetry or customer logs shall vest jointly in the Company and Counterparty, with Counterparty retaining an irrevocable, worldwide, royalty-free, sublicensable right to commercialize such Joint IP without accounting to the Company.

Section {art_num}.02 Open Source Software & Copyleft Contamination.
Set forth on Schedule {art_num}.02 is a complete and accurate list of all Open Source Software (including AGPL-3.0, GPL-3.0, and SSPLv1 licensed components) linked, statically or dynamically, or otherwise incorporated into the Company's core SaaS platform. Note that Module {mod_code} of the distributed query engine incorporates code licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) and is interacted with by external enterprise customers over a computer network without a commercial dual-license exception.

Section {art_num}.03 Training Data Provenance and Copyright Indemnification Carve-Out.
Notwithstanding anything to the contrary in Article IX, the Sellers make no representation or warranty that the web-scraped corpus comprising Dataset Batch #{batch_id} ({token_count} billion tokens) complies with third-party terms of service or the EU Artificial Intelligence Act (Regulation (EU) 2024/1689) transparency obligations, and all third-party copyright infringement claims arising out of or relating to Dataset Batch #{batch_id} are expressly excluded from Seller's Special Indemnification Obligations.""",

    """ARTICLE {art_num}. CHANGE OF CONTROL, ASSIGNMENT & COMMERCIAL EXCLUSIVITY
Section {art_num}.01 Anti-Assignment and Deemed Change of Control.
Neither this Master Enterprise Agreement (Contract Ref #{contract_ref}) nor any rights, licenses, or obligations hereunder may be assigned, delegated, or transferred by Vendor, whether voluntarily, by operation of law, or otherwise, without the prior written consent of Customer ({counterparty}), which consent may be withheld, conditioned, or delayed in Customer's sole and absolute discretion. For purposes of this Section {art_num}.01, any merger, consolidation, reorganization, recapitalization, sale of all or substantially all assets, or direct or indirect transfer of more than twenty percent (20%) of the voting securities or governance control of Vendor shall be deemed a prohibited assignment ("Deemed Assignment"). Upon any Deemed Assignment without Customer's prior written consent:
  (a) Customer may immediately terminate this Agreement for cause without penalty;
  (b) Vendor shall immediately refund to Customer one hundred fifty percent (150%) of all prepaid unearned annual recurring revenue (ARR) fees (currently ${arr_amount} million); and
  (c) Customer's source code escrow release trigger under Exhibit E shall automatically vest, granting Customer a perpetual, royalty-free license to the escrowed source code.

Section {art_num}.02 Most Favored Nation (MFN) and Sector Exclusivity.
During the Initial Term and any Renewal Term, Vendor shall not, and shall cause its Affiliates (including any future parent entity, acquirer, or sister subsidiary following a Change of Control) not to, offer, license, or provide any competing machine-learning analytics or automated compliance services to any Top-50 Global Financial Institution listed on Annex {art_num}-B, or offer pricing terms to any third party more favorable than the unit rates set forth in Pricing Schedule 1 without retroactively crediting Customer the difference back to the Effective Date.""",

    """ARTICLE {art_num}. INDEMNIFICATION, LIMITATION OF LIABILITY & ESCROW RECOURSE
Section {art_num}.01 Indemnification by the Sellers.
Subject to the limitations set forth in this Article {art_num}, from and after the Closing, the Sellers shall, severally and not jointly (in accordance with their respective Pro Rata Shares), indemnify, defend and hold harmless Buyer and its Affiliates from and against any and all Losses actually suffered or incurred by Buyer arising out of: (i) any breach of any representation or warranty of the Company contained in Article III; (ii) any breach of any covenant of the Company requiring performance prior to Closing; and (iii) Pre-Closing Taxes.

Section {art_num}.02 Tipping Basket, Cap and Exclusive Remedy Limitations.
(a) Deductible / Basket: Sellers shall not be liable for any Losses under Section {art_num}.01(i) unless and until the aggregate amount of all such Losses exceeds ${basket_musd} million (the "Deductible"), in which event Sellers shall only be liable for Losses in excess of the Deductible; provided, further, that no individual claim (or series of related claims) resulting in Losses of less than ${mini_basket_kusd},000 (the "De Minimis Threshold") shall be counted toward the Deductible.
(b) General Cap: The maximum aggregate liability of the Sellers for all Losses under Section {art_num}.01(i) (INCLUDING breaches of Intellectual Property representations under Section 4.01 and Data Privacy representations under Section 4.08) shall not exceed the Indemnity Escrow Amount of ${cap_musd} million (representing {cap_pct}% of the Base Purchase Price).
(c) Survival: All representations and warranties (including Fundamental Representations and Tax Representations) shall survive the Closing for a period of nine (9) months and shall thereafter terminate and be of no further force or effect.
(d) Consequential Damages Waiver: IN NO EVENT SHALL ANY PARTY BE LIABLE FOR ANY INCIDENTAL, INDIRECT, SPECIAL, PUNITIVE, EXEMPLARY, OR CONSEQUENTIAL DAMAGES, INCLUDING LOSS OF REVENUE, DIMINUTION IN VALUE, OR MULTIPLES OF EARNINGS, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.""",

    """ARTICLE {art_num}. DATA PRIVACY, CROSS-BORDER TRANSFERS & CYBERSECURITY INCIDENT HISTORY
Section {art_num}.01 Processing of Regulated Personal Data and PHI.
In connection with the performance of the Services under Addendum #{contract_ref}, Company processes Protected Health Information ("PHI") subject to HIPAA/HITECH, biometric identifiers subject to the Illinois Biometric Information Privacy Act (740 ILCS 14/1 et seq. ("BIPA")), and European Economic Area personal data subject to Regulation (EU) 2016/679 ("GDPR").
(a) Subprocessors: Company currently utilizes {subproc_count} third-party subprocessors located in jurisdictions without an European Commission adequacy decision, without executed Standard Contractual Clauses (Module Three: Processor-to-Processor) or Transfer Impact Assessments on file.
(b) Historical Security Incident #{incident_id}: On {jda_date}, an unauthorized actor accessed production S3 storage bucket `prod-legal-vault-{incident_id}` containing {breach_records},000 unencrypted customer records. Remediation was performed internally; no notification was provided to affected data subjects, state Attorneys General, or supervisory authorities within the 72-hour window prescribed by GDPR Article 33, based on internal counsel's preliminary determination that exfiltration could not be conclusively proven from netflow logs.
(c) Unlimited Liability Exception: Under Customer DPA #{contract_ref}, Vendor's liability for any breach of data protection obligations, unauthorized disclosure of Customer Data, or violation of BIPA/GDPR is UNLIMITED and expressly carved out from all contractual caps on liability.""",

    """ARTICLE {art_num}. CREDIT FACILITY COVENANTS, CROSS-DEFAULT & LITIGATION CONTINGENCIES
Section {art_num}.01 Senior Secured Credit Facility Covenants.
Reference is made to that certain Amended and Restated Credit Agreement dated {jda_date} among the Company, the Lenders party thereto, and {counterparty} as Administrative Agent, providing for a ${loan_musd} million Senior Secured Term Loan Facility.
(a) Financial Covenants: The Company must maintain at the end of each fiscal quarter (i) a Maximum Consolidated Total Net Leverage Ratio not to exceed 3.50 to 1.00 (current Q3 actual: 3.48 to 1.00), and (ii) Minimum Liquidity of at least ${min_liq_musd} million in unrestricted cash held at Administrative Agent.
(b) Cross-Default: Any default or termination right triggered under any Material Customer Contract representing more than 5% of Consolidated EBITDA (which includes Contract Ref #{contract_ref}) constitutes an immediate Event of Default under Section 8.01(e) of the Credit Agreement, permitting Administrative Agent to accelerate all outstanding principal and accrued interest immediately without notice or cure period.

Section {art_num}.02 Pending and Threatened Litigation (Schedule {art_num}.02).
(a) *Apex Neural Systems, Inc. v. Target Corp.* (U.S. District Court for the District of Delaware, Case No. 1:26-cv-0{incident_id}): Plaintiff alleges willful infringement of U.S. Patent Nos. 11,482,{mod_code} and 11,901,{batch_id} covering sparse attention KV-cache routing. Plaintiff seeks permanent injunctive relief against the Company's core inference product and treble damages estimated by Plaintiff's damages expert at ${patent_dmg_musd} million. Fact discovery closes in 45 days; Inter Partes Review (IPR) institution was denied by the PTAB on {jda_date}.
(b) Former Executive Wage & Equity Dispute: Former Chief Scientist Dr. R. Vance asserts claims for promised but ungranted 1.75% fully-diluted anti-dilution equity promised via informal email thread dated {jda_date}, seeking constructive trust over Company IP orrescission of his Invention Assignment Agreement (which omitted the California Labor Code § 2870 statutory notice)."""
]

COUNTERPARTIES = [
    "Example Global Banking Corp.",
    "Contoso Healthcare Systems, LLC",
    "Northwind Defense Technologies, Inc.",
    "Fabrikam Financial Services AG",
    "Litware Semiconductor Holdings Ltd.",
    "AdventureWorks Cloud Infrastructure Corp.",
    "Tailspin Insurance Syndicate",
    "Woodgrove National Bancorp, N.A.",
]

DATES = [
    "January 14, 2024", "March 29, 2024", "July 19, 2024",
    "November 4, 2024", "February 11, 2025", "June 30, 2025",
    "September 22, 2025", "January 18, 2026", "May 12, 2026"
]

REFERENCE_LEGAL_OUTPUT = """### 1. EXECUTIVE RISK SUMMARY & MATERIALITY MATRIX

| Risk ID | Severity | Source Clause | Exposure Category | Quantified Financial / Operational Impact |
| :--- | :--- | :--- | :--- | :--- |
| **R-01** | **CRITICAL** | Art. 3.01 / 9.01 | Change-of-Control Deemed Assignment & 150% ARR Refund Penalty | Immediate contract termination upon >20% equity transfer; mandatory **150% prepaid ARR refund**, automatic **source code escrow release** to Customer, and **Cross-Default acceleration** under Senior Credit Facility. |
| **R-02** | **CRITICAL** | Art. 3.02 | Affiliate-Wide MFN & Top-50 Financial Sector Exclusivity | Binds Acquirer and all post-closing Affiliates globally; prohibits servicing Top-50 financial institutions and forces retroactive MFN price reductions across Acquirer's existing enterprise base. |
| **R-03** | **CRITICAL** | Art. 2.02 / 2.03 | AGPL-3.0 Network Copyleft Contamination & Excluded AI Training Data | Network-facing SaaS module incorporates AGPL-3.0 code without commercial exception, exposing proprietary stack to mandatory source disclosure; scraped training corpus carries **zero Seller indemnification**. |
| **R-04** | **HIGH** | Art. 5.01(b)-(c) | Unreported Data Breach, Missing GDPR SCCs & BIPA Unlimited Liability | Prior breach of unencrypted records went unreported under GDPR Art. 33 / state breach laws; missing Processor-to-Processor SCCs; Customer DPA imposes **unlimited liability** on Target while Sellers cap buyer recourse. |
| **R-05** | **HIGH** | Art. 4.02 | Inadequate Indemnity Cap, Tipping/Deductible Mismatch & 9-Month Survival | True deductible excludes IP/Privacy claims from special caps; 9-month survival cuts off tax, IP, and fundamental claims prematurely; consequential damages waiver blocks diminution-in-value recovery. |
| **R-06** | **HIGH** | Art. 6.02 | Willful Patent Infringement (PTAB IPR Denied) & Unassigned Founder IP | Active Delaware patent suit seeking injunction + treble damages after PTAB denied IPR; Former Chief Scientist's Invention Assignment omitted Cal. Lab. Code § 2870 notice, threatening core IP title. |

---

### 2. CHANGE-OF-CONTROL, ASSIGNMENT & TERMINATION TRIGGERS
1. **Deemed Assignment at 20% Threshold:** Unlike market-standard 50% voting control thresholds, Section 3.01 triggers a prohibited assignment upon any direct or indirect transfer exceeding **20% of voting securities**.
2. **Punitive Liquidated Consequences:** Closing without prior written consent triggers (a) immediate termination for cause, (b) a **150% refund of unearned ARR** (an unenforceable penalty disguised as liquidated damages, yet high litigation risk), and (c) **automatic release of escrowed source code** under Exhibit E.
3. **Affiliate-Binding Exclusivity & MFN Contamination:** Section 3.02 explicitly extends non-compete and MFN pricing obligations to *"any future parent entity, acquirer, or sister subsidiary following a Change of Control."* If Acquirer closes without a written waiver, Acquirer's own pre-existing business lines become subject to Annex B sector restrictions.
4. **Debt Cross-Default Cascade:** Termination of any Material Customer Contract (>5% Consolidated EBITDA) triggers an immediate Event of Default under Section 8.01(e) of the Senior Secured Credit Agreement, accelerating the entire Term Loan facility with current leverage at 3.48x against a 3.50x covenant ceiling.

---

### 3. INDEMNIFICATION, LIABILITY CAPS & SURVIVAL ANALYSIS
1. **Asymmetric Liability Trap:** Under the Customer DPAs (Section 5.01(c)), the Target has accepted **unlimited liability** for GDPR/BIPA/data breach claims, whereas under the acquisition agreement (Section 4.02(b)), Sellers cap their indemnification for IP and Data Privacy breaches at the general Escrow Amount. Buyer absorbs 100% of the tail risk above escrow.
2. **Survival Period Deficiencies:** A flat **9-month survival period** for Fundamental Representations, IP Ownership, and Pre-Closing Taxes violates market practice under Delaware law (*10 Del. C. § 8106* permits up to 3 years, or 20 years for contracts under seal, and Tax claims must survive until 60 days post-statute of limitations).
3. **Damages Waiver Overreach:** Section 4.02(d) waives "diminution in value" and "multiples of earnings," which under Delaware jurisprudence (* Cobalt Operating, LLC v. James Crystal Enterprises, LLC*) deprives Buyer of its primary measure of general damages when EBITDA-generating contracts are impaired.

---

### 4. INTELLECTUAL PROPERTY, DATA PRIVACY & REGULATORY COMPLIANCE
1. **Joint IP Leakage:** Under the Joint Development Agreement (Section 2.01), derivative architectures and fine-tuned adapters trained on Counterparty telemetry vest jointly with an unrestricted, royalty-free right for Counterparty to sublicense to Target's direct competitors.
2. **AGPL-3.0 Remediation Condition Precedent:** Dynamic/network interaction with AGPL-3.0 code in the query engine triggers Section 13 of AGPL-3.0, requiring complete corresponding source code distribution to all network users unless isolated via clean-room microservice replacement prior to Closing.
3. **BIPA & GDPR Statutory Exposure:** Unreported incident involving unencrypted records combined with BIPA processing ($1,000 per negligent violation / $5,000 per reckless violation per scan under *Cothron v. White Castle*) requires a dedicated Special Escrow.

---

### 5. BINDING REDLINE AMENDMENTS & REMEDIATION LANGUAGE

#### A. Amendment to Section 4.02(b)-(c) (Indemnification Caps & Survival)
> "(b) General Cap; Fundamental & Special Indemnities: The maximum aggregate liability of the Sellers for Losses under Section 4.01(i) [DELETED: (INCLUDING breaches of Intellectual Property representations under Section 4.01 and Data Privacy representations under Section 4.08)] shall not exceed the Indemnity Escrow Amount; <<INSERTED: provided, however, that the Deductible and the General Cap shall NOT apply to (i) breaches of Fundamental Representations (which shall be capped at 100% of the Aggregate Purchase Price), (ii) breaches of Section 2.01-2.03 (Intellectual Property & AI Assets) and Section 5.01 (Data Privacy & Cybersecurity), which shall be subject to a Special IP/Privacy Cap equal to fifty percent (50%) of the Aggregate Purchase Price, or (iii) Fraud or Willful Misconduct.>>
> (c) Survival: General representations shall survive for [DELETED: nine (9)] <<INSERTED: eighteen (18)>> months; <<INSERTED: IP and Data Privacy representations shall survive for thirty-six (36) months; and Fundamental Representations and Pre-Closing Tax Indemnities shall survive until sixty (60) days following the expiration of the applicable statute of limitations (giving effect to any extensions thereof).>>"

#### B. Amendment to Section 3.02 (Customer Waiver Letter — MFN & Exclusivity Containment)
> "Notwithstanding anything to the contrary in Section 3.02 of the Master Enterprise Agreement, Customer hereby consents to the acquisition of Vendor by Acquirer and agrees that: (i) the Change of Control shall not constitute a Deemed Assignment, trigger any refund of prepaid ARR, or release any escrowed materials under Exhibit E; and (ii) the exclusivity and Most Favored Nation covenants in Section 3.02 [DELETED: shall bind Vendor and its Affiliates (including any future parent entity, acquirer, or sister subsidiary)] <<INSERTED: shall apply solely to the standalone Legacy Vendor Product SKU as existing immediately prior to the Closing Date, and shall in no event apply to, restrict, or bind Acquirer or any of Acquirer's existing or future Affiliates, products, or customer agreements.>>"
"""


def build_legal_document(seed: int, target_input_tokens: int) -> str:
    rng = random.Random(seed)
    # Approx 1 token ~= 4.0 chars in English legal text
    target_chars = int(target_input_tokens * 3.85)
    header = (
        f"CONFIDENTIAL M&A DUE DILIGENCE PACKET — DEAL FILE #MNA-2026-{1000 + seed}\n"
        f"MATTER: Acquisition of NeuralLex Enterprise Systems, Inc. ('Target') by Acquirer Corp.\n"
        f"INSTRUCTIONS TO COUNSEL: Review all attached Deal Exhibits, Material Contracts, Debt Instruments, "
        f"and Litigation Schedules below. Provide the complete 5-section Legal Review & Redline Memorandum.\n"
        f"{'=' * 92}\n\n"
    )
    chunks = [header]
    current_len = len(header)
    art_num = 1
    while current_len < target_chars:
        tmpl = CLAUSE_TEMPLATES[(art_num + seed) % len(CLAUSE_TEMPLATES)]
        text = tmpl.format(
            art_num=art_num,
            mae_thresh=rng.choice([5, 10, 15, 25]),
            auth_common=rng.choice([100, 150, 200]),
            auth_pref=rng.choice([25, 50, 75]),
            issued_common=rng.randint(40, 90),
            issued_pref=rng.randint(10, 24),
            options_pool=rng.randint(5, 18),
            warrants=rng.randint(1, 5),
            jda_date=rng.choice(DATES),
            counterparty=rng.choice(COUNTERPARTIES),
            mod_code=rng.randint(104, 998),
            batch_id=rng.randint(401, 899),
            token_count=rng.choice([120, 250, 480, 750]),
            contract_ref=f"MSA-{rng.randint(2022, 2026)}-{rng.randint(100, 999)}",
            arr_amount=rng.choice([4.5, 8.2, 12.0, 18.5, 24.0]),
            basket_musd=rng.choice([2.5, 4.0, 5.0, 7.5]),
            mini_basket_kusd=rng.choice([150, 250, 500]),
            cap_musd=rng.choice([12.5, 15.0, 20.0, 25.0]),
            cap_pct=rng.choice([5.0, 7.5, 10.0]),
            subproc_count=rng.randint(7, 23),
            incident_id=rng.randint(1004, 9870),
            breach_records=rng.choice([45, 120, 310, 850]),
            loan_musd=rng.choice([75, 120, 180, 250]),
            min_liq_musd=rng.choice([15, 25, 40]),
            patent_dmg_musd=rng.choice([35, 60, 95, 140]),
        )
        block = f"--- EXHIBIT {art_num:03d} ---\n{text}\n\n"
        chunks.append(block)
        current_len += len(block)
        art_num += 1

    full_doc = "".join(chunks)
    if len(full_doc) > target_chars:
        full_doc = full_doc[:target_chars]
    footer = (
        "\n\n[END OF DUE DILIGENCE DATA ROOM PACKET]\n"
        "TASK FOR PRINCIPAL LEGAL COUNSEL: Based on the system instructions and all exhibits above, "
        "generate the complete 5-part Legal Review & Redline Memorandum (1. Executive Risk Summary & Materiality Matrix, "
        "2. Change-of-Control & Termination Triggers, 3. Indemnification & Liability Cap Analysis, "
        "4. IP, Privacy & Regulatory Compliance, and 5. Binding Redline Amendments)."
    )
    return full_doc + footer


def generate_jsonl(output_path: Path, num_prompts: int, input_tokens: int, output_tokens: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(num_prompts):
            user_prompt = build_legal_document(seed=1000 + i, target_input_tokens=input_tokens)
            combined_prompt = (
                f"System Instructions:\n{LEGAL_SYSTEM_PROMPT}\n\n"
                f"User Legal Matter Submission:\n{user_prompt}"
            )
            record = {
                "prompt": combined_prompt,
                "output_tokens": output_tokens,
                "system_prompt": LEGAL_SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "reference_output": REFERENCE_LEGAL_OUTPUT,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Generated {num_prompts} legal prompts (~{input_tokens} input tokens, {output_tokens} output tokens) -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Legal M&A Due Diligence benchmark dataset for vLLM")
    parser.add_argument("--output-dir", type=str, default="datasets", help="Directory to write JSONL datasets")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    # 1. High-throughput commercial contract review (4K input, 1024 output, 128 prompts)
    generate_jsonl(out_dir / "legal_contract_4k_in_1k_out.jsonl", num_prompts=128, input_tokens=4096, output_tokens=1024)
    # 2. Multi-agreement M&A due diligence (16K input, 1024 output, 64 prompts)
    generate_jsonl(out_dir / "legal_mna_16k_in_1k_out.jsonl", num_prompts=64, input_tokens=16384, output_tokens=1024)
    # 3. Deep Data Room review (64K input, 1024 output, 32 prompts)
    generate_jsonl(out_dir / "legal_dataroom_64k_in_1k_out.jsonl", num_prompts=32, input_tokens=65536, output_tokens=1024)
    # 4. Ultra-long litigation discovery & full deal packet (128K input, 1024 output, 16 prompts)
    generate_jsonl(out_dir / "legal_discovery_128k_in_1k_out.jsonl", num_prompts=16, input_tokens=131072, output_tokens=1024)
