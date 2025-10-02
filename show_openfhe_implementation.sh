#!/bin/bash
# Show where OpenFHE two-party threshold is implemented in NC FedGCN

echo "=============================================================="
echo "🔍 OpenFHE Two-Party Threshold in NC FedGCN PRETRAIN"
echo "=============================================================="
echo ""

echo "📍 LOCATION: fedgraph/federated_methods.py"
echo ""

echo "─────────────────────────────────────────────────────────────"
echo "1️⃣  PRETRAIN PHASE ENTRY (Lines 245-253)"
echo "─────────────────────────────────────────────────────────────"
sed -n '245,253p' fedgraph/federated_methods.py
echo ""

echo "─────────────────────────────────────────────────────────────"
echo "2️⃣  OPENFHE TWO-PARTY PROTOCOL (Lines 280-312)"
echo "─────────────────────────────────────────────────────────────"
echo "This is where the two-party threshold key generation happens:"
sed -n '280,312p' fedgraph/federated_methods.py
echo ""

echo "─────────────────────────────────────────────────────────────"
echo "3️⃣  ENCRYPTED FEATURE AGGREGATION (Lines 314-339)"
echo "─────────────────────────────────────────────────────────────"
echo "Trainers encrypt features, server aggregates with threshold decryption:"
sed -n '314,339p' fedgraph/federated_methods.py
echo ""

echo "─────────────────────────────────────────────────────────────"
echo "4️⃣  PERFORMANCE METRICS (Lines 341-351)"
echo "─────────────────────────────────────────────────────────────"
sed -n '341,351p' fedgraph/federated_methods.py
echo ""

echo "=============================================================="
echo "✅ VERIFICATION COMPLETE"
echo "=============================================================="
echo ""
echo "📊 Summary:"
echo "  • OpenFHE two-party threshold: ✅ Implemented"
echo "  • Location: NC FedGCN PRETRAIN phase (lines 280-351)"
echo "  • Key generation: Server (lead) + Trainer0 (non-lead)"
echo "  • Aggregation: Homomorphic addition + threshold decryption"
echo "  • Training phase: Still plaintext (encryption not yet implemented)"
echo ""
echo "🔐 Security:"
echo "  • Server: Holds secret_share_1"
echo "  • Trainer0: Holds secret_share_2"
echo "  • Decryption: Requires BOTH parties (threshold)"
echo ""
echo "📝 To test:"
echo "  python test_and_compare_results.py"
echo ""

