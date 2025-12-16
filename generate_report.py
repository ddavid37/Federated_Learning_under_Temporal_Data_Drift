"""
PDF Report Generator for FL Under Concept Drift
Generates a comprehensive academic-style report with all experimental results.
"""

import os
from fpdf import FPDF

# --- Configuration ---
REPORT_FILE = "Final_Project_Report.pdf"

# Image files
IMG_HEATMAP_2 = "heatmap_case2.png"
IMG_HEATMAP_3 = "heatmap_case3.png"
IMG_RADAR = "radar_robustness.png"
IMG_RECOVERY = "recovery_curve.png"
IMG_ABLATION = "ablation_buffer_size.png"

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 10)
        self.cell(0, 10, 'NNDL Project: Federated Learning Under Concept Drift', 0, 1, 'R')
        self.line(10, 18, 200, 18)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(240, 240, 245)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Times', '', 11)
        self.multi_cell(0, 6, body)
        self.ln()
    
    def section_header(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, title, 0, 1, 'L')
    
    def add_table_row(self, cols, widths, bold=False):
        if bold:
            self.set_font('Arial', 'B', 10)
        else:
            self.set_font('Times', '', 10)
        for i, col in enumerate(cols):
            self.cell(widths[i], 7, str(col), 1, 0, 'C')
        self.ln()

def generate_pdf():
    pdf = PDF()
    
    # ==================== PAGE 1: Title & Abstract ====================
    pdf.add_page()
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 15, "Federated Learning Under Concept Drift:", 0, 1, 'C')
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 10, "Experience Replay as a Mitigation Strategy", 0, 1, 'C')
    pdf.ln(8)
    
    pdf.chapter_title("Abstract")
    pdf.chapter_body(
        "We investigate the problem of concept drift in Federated Learning (FL) using Fashion-MNIST as a testbed. "
        "Standard FedAvg suffers from catastrophic forgetting when client data distributions shift over time, "
        "dropping from 74% to approximately 28% accuracy in our seasonal drift simulation. We evaluate "
        "client-side experience replay buffers as a mitigation strategy. "
        "Our experiments show that experience replay with a 50-sample-per-class buffer recovers to 78-82% accuracy. "
        "We provide detailed experimental protocols, ablation studies on buffer size, and discuss limitations of our approach."
    )
    
    # ==================== PAGE 2: Introduction & Related Work ====================
    pdf.add_page()
    pdf.chapter_title("1. Introduction")
    pdf.chapter_body(
        "Federated Learning enables collaborative model training without centralizing raw data [1]. However, "
        "real-world deployments face concept drift - where data distributions evolve over time (e.g., seasonal "
        "fashion trends, changing user preferences). This non-stationarity causes catastrophic forgetting, "
        "where models trained on new distributions lose performance on previously learned patterns.\n\n"
        "This project addresses: How can we maintain model performance in FL under temporal concept drift?\n\n"
        "Our contributions:\n"
        "- Empirical demonstration of catastrophic forgetting in FedAvg under simulated seasonal drift\n"
        "- Implementation and evaluation of client-side experience replay as a mitigation strategy\n"
        "- Baseline comparison with FedAvg on IID data to isolate drift effects\n"
        "- Ablation study on replay buffer size"
    )
    
    pdf.chapter_title("2. Related Work")
    pdf.chapter_body(
        "Federated Learning: McMahan et al. [1] introduced FedAvg for distributed training. Privacy benefits "
        "stem from keeping raw data local; however, gradient updates may still leak information without "
        "additional protections like differential privacy [2] or secure aggregation.\n\n"
        "Continual Learning: Experience replay is a core technique from continual learning [3], where a memory "
        "buffer stores representative samples from previous tasks. This has been applied in centralized settings "
        "to mitigate forgetting.\n\n"
        "FL under Non-Stationarity: Recent work addresses drift in FL through multi-model approaches [4], "
        "clustering clients by distribution similarity, and adaptive aggregation. Our work combines client-local "
        "replay buffers with standard FedAvg, requiring no server-side modifications.\n\n"
        "Limitations of Our Literature Review: We acknowledge that FL+continual learning is an active area, "
        "and more comprehensive surveys exist. We focus on foundational methods for this project scope."
    )
    
    # ==================== PAGE 3: Experimental Setup ====================
    pdf.add_page()
    pdf.chapter_title("3. Experimental Setup")
    
    pdf.section_header("3.1 Dataset & Preprocessing")
    pdf.chapter_body(
        "Dataset: Fashion-MNIST (60,000 training, 10,000 test images, 28x28 grayscale)\n"
        "Classes: T-shirt(0), Trouser(1), Pullover(2), Dress(3), Coat(4), Sandal(5), Shirt(6), Sneaker(7), Bag(8), Ankle-boot(9)\n"
        "Preprocessing: Pixel values normalized to [0, 1]. No data augmentation applied."
    )
    
    pdf.section_header("3.2 Client Configuration")
    pdf.chapter_body(
        "Number of Clients: 10 (all participate each round, i.e., full participation)\n"
        "Data Distribution:\n"
        "  - Phase 0 (Init): 10,000 IID samples split equally (1,000/client)\n"
        "  - Seasonal Phases: Non-IID, class-restricted (see drift schedule below)\n"
        "Client Selection: All 10 clients participate every round (no random sampling)"
    )
    
    pdf.section_header("3.3 Seasonal Drift Schedule")
    pdf.chapter_body(
        "We simulate temporal concept drift by restricting available classes per season:\n\n"
        "Phase 0 (Init IID): All 10 classes, balanced - 5 rounds\n"
        "Phase 1 (Winter): Classes [4-Coat, 9-Ankle boot, 2-Pullover] only - 5 rounds\n"
        "Phase 2 (Spring): Classes [1-Trouser, 6-Shirt, 8-Bag] only - 5 rounds\n"
        "Phase 3 (Summer): Classes [0-T-shirt, 3-Dress, 5-Sandal] only - 5 rounds\n"
        "Phase 4 (Fall): Classes [7-Sneaker, 2-Pullover, 6-Shirt, 1-Trouser] - 5 rounds\n\n"
        "Total: 25 communication rounds. Each phase transition represents abrupt drift."
    )
    
    pdf.section_header("3.4 Model Architecture")
    pdf.chapter_body(
        "CNN Architecture (same for all experiments):\n"
        "  - Conv2d(1->32, 5x5, padding=2) + ReLU + MaxPool(2x2)\n"
        "  - Conv2d(32->64, 5x5, padding=2) + ReLU + MaxPool(2x2)\n"
        "  - Flatten -> Linear(3136->512) + ReLU -> Linear(512->10)\n"
        "Total Parameters: ~1.7M"
    )
    
    pdf.section_header("3.5 Training Hyperparameters")
    pdf.chapter_body(
        "Federated Settings:\n"
        "  - Communication Rounds: 25 total (5 per phase)\n"
        "  - Local Epochs: 1 per round\n"
        "  - Local Batch Size: 32\n"
        "  - Optimizer: SGD with momentum=0.9, lr=0.01\n"
        "  - Aggregation: FedAvg (simple weight averaging)\n\n"
        "Centralized Baseline:\n"
        "  - Epochs: 10\n"
        "  - Batch Size: 64\n"
        "  - Optimizer: Adam, lr=0.001"
    )
    
    pdf.section_header("3.6 Experience Replay Configuration")
    pdf.chapter_body(
        "Buffer Location: Client-side (each client maintains own buffer)\n"
        "Buffer Update Policy: Fill-up (add samples until capacity, then stop)\n"
        "Buffer Size: 50 samples per class per client (default)\n"
        "Total Buffer Memory: 50 * 10 classes * 784 bytes = ~392KB per client\n"
        "Replay Strategy: Concatenate buffer with current season data each round\n"
        "Sampling Ratio: No explicit ratio; all buffered samples used alongside new data"
    )
    
    pdf.section_header("3.7 Evaluation Protocol")
    pdf.chapter_body(
        "Test Set: Full Fashion-MNIST test set (10,000 samples, all classes)\n"
        "Evaluation Frequency: After every communication round\n"
        "Metrics: Global accuracy, per-class accuracy\n"
        "Note: We evaluate on ALL classes even when training on subset (measures forgetting)"
    )

    # ==================== PAGE 4: Results ====================
    pdf.add_page()
    pdf.chapter_title("4. Results")
    
    pdf.section_header("4.1 Global Accuracy Over Time")
    if os.path.exists(IMG_RECOVERY):
        pdf.image(IMG_RECOVERY, x=10, w=190)
        pdf.ln(5)

    pdf.chapter_body(
        "Key Observations:\n"
        "- Centralized Baseline: ~88% (trained on IID data only, upper bound reference)\n"
        "- FedAvg on IID (no drift): Maintains ~75-80% accuracy throughout 25 rounds\n"
        "- FedAvg (with drift, Rounds 1-5): Reaches ~74% accuracy on IID data\n"
        "- FedAvg (with drift, Rounds 6+): Drops to 26-36% as forgetting occurs\n"
        "- FedAvg + Replay: Maintains 66-82% throughout, recovering after each drift\n\n"
        "The accuracy drop in standard FedAvg from 74% to ~28% (a 46 percentage point drop) demonstrates "
        "catastrophic forgetting. The IID baseline proves this is due to drift, not FL itself. "
        "Experience replay reduces this drop significantly."
    )

    # ==================== PAGE 5: Heatmaps ====================
    pdf.add_page()
    pdf.chapter_title("4.2 Per-Class Accuracy Analysis")
    
    pdf.section_header("Standard FedAvg (Catastrophic Forgetting)")
    if os.path.exists(IMG_HEATMAP_2):
        pdf.image(IMG_HEATMAP_2, x=15, w=180)
        pdf.ln(2)
    pdf.chapter_body(
        "The heatmap shows complete knowledge loss for classes not in current season. "
        "During Summer (Rounds 16-20), only T-shirt, Dress, and Sandal retain accuracy; "
        "all other classes drop to 0%. Black vertical lines mark phase transitions."
    )

    pdf.add_page()
    pdf.section_header("FedAvg + Experience Replay (Forgetting Mitigated)")
    if os.path.exists(IMG_HEATMAP_3):
        pdf.image(IMG_HEATMAP_3, x=15, w=180)
        pdf.ln(2)
    pdf.chapter_body(
        "With replay, classes maintain non-zero accuracy across all rounds. While current-season "
        "classes show highest performance, buffered samples prevent complete forgetting of other classes. "
        "Final round shows balanced performance across most categories."
    )

    # ==================== PAGE 6: Radar & Ablation ====================
    pdf.add_page()
    pdf.chapter_title("4.3 Robustness Profile")
    
    if os.path.exists(IMG_RADAR):
        pdf.image(IMG_RADAR, x=40, w=130)
        pdf.ln(5)
    
    pdf.chapter_body(
        "The radar chart shows final-round per-class accuracy for each approach. "
        "FedAvg+Replay (green) achieves more uniform coverage compared to standard FedAvg (red), "
        "which shows gaps for forgotten classes. The IID baseline (orange) shows what FL achieves without drift."
    )
    
    pdf.add_page()
    pdf.section_header("4.4 Buffer Size Ablation")
    
    if os.path.exists(IMG_ABLATION):
        pdf.image(IMG_ABLATION, x=10, w=190)
        pdf.ln(5)
    
    pdf.chapter_body(
        "We conducted an ablation study varying the replay buffer size:\n\n"
        "Buffer Size (per class) | Final Accuracy | Memory/Client\n"
        "         0 (no replay)  |     ~28%       |      0 KB\n"
        "        10 samples      |     ~65%       |     78 KB\n"
        "        25 samples      |     ~74%       |    196 KB\n"
        "        50 samples      |     ~78%       |    392 KB\n"
        "       100 samples      |     ~80%       |    784 KB\n\n"
        "Observations: Even a small buffer (10/class) substantially improves over no replay. "
        "Returns diminish beyond 50 samples/class for this dataset size."
    )

    # ==================== PAGE 7: Discussion ====================
    pdf.add_page()
    pdf.chapter_title("5. Discussion")
    
    pdf.section_header("5.1 Why Experience Replay Works")
    pdf.chapter_body(
        "Experience replay maintains a diverse training signal across all classes, even when "
        "current season data is restricted. This prevents the model from overwriting features "
        "for classes not present in the current phase.\n\n"
        "Key insight: The buffer acts as a 'memory' that reminds the model of previously seen "
        "patterns during each local training step."
    )
    
    pdf.section_header("5.2 Privacy Considerations")
    pdf.chapter_body(
        "In our implementation, raw data does not leave client devices - only model weight updates "
        "are transmitted. However, we note important caveats:\n\n"
        "- We do NOT implement differential privacy or secure aggregation\n"
        "- Model updates may leak information about training data (gradient leakage attacks [2])\n"
        "- The replay buffer stores raw samples locally, which is acceptable under FL assumptions "
        "  but increases local storage requirements\n\n"
        "For production deployments requiring stronger privacy guarantees, differential privacy "
        "(adding noise to updates) or secure aggregation protocols should be considered."
    )
    
    pdf.section_header("5.3 Communication Overhead")
    pdf.chapter_body(
        "Per-round communication for our CNN:\n"
        "- Model size: ~1.7M parameters x 4 bytes = ~6.8 MB per client per round\n"
        "- With 10 clients and 25 rounds: ~1.7 GB total communication\n\n"
        "This is substantially less than transmitting raw Fashion-MNIST images, but the comparison "
        "depends heavily on model size and data volume."
    )

    # ==================== PAGE 8: Limitations ====================
    pdf.add_page()
    pdf.chapter_title("6. Limitations")
    pdf.chapter_body(
        "1. Simulated Drift: Our seasonal shifts are synthetic and abrupt. Real-world drift is "
        "often gradual and harder to detect.\n\n"
        "2. Small Scale: 10 clients with full participation is not representative of "
        "cross-device FL (thousands of clients, partial participation).\n\n"
        "3. Simple Buffer Policy: We use fill-up (first-come-first-serve). Reservoir sampling "
        "or importance-weighted selection might perform better.\n\n"
        "4. Single Dataset: Fashion-MNIST is a controlled benchmark. Results may differ on "
        "more complex datasets or tasks.\n\n"
        "5. No Drift Detection: We manually schedule drift. Automatic drift detection would "
        "be needed for real deployments.\n\n"
        "6. Privacy Analysis: We did not evaluate gradient leakage risks or implement DP."
    )
    
    pdf.chapter_title("7. Future Work")
    pdf.chapter_body(
        "- Evaluate on CIFAR-10/100 with more realistic drift patterns\n"
        "- Implement differential privacy and measure accuracy/privacy trade-offs\n"
        "- Compare with other continual learning methods (EWC, PackNet)\n"
        "- Scale to larger client populations with partial participation\n"
        "- Automatic drift detection mechanisms"
    )

    # ==================== PAGE 9: Conclusion & References ====================
    pdf.add_page()
    pdf.chapter_title("8. Conclusion")
    pdf.chapter_body(
        "We demonstrated that standard FedAvg experiences severe catastrophic forgetting under "
        "simulated concept drift, with accuracy dropping from 74% to 28% on Fashion-MNIST. "
        "Our IID baseline experiment confirms this collapse is due to drift, not FL itself.\n\n"
        "Client-side experience replay with a 50-sample-per-class buffer effectively mitigates "
        "forgetting, recovering to 78-82% accuracy while maintaining a single global model.\n\n"
        "Key Takeaway: For seasonal or temporal drift with overlapping class structures, "
        "experience replay provides a simple, effective solution compatible with standard "
        "FL infrastructure, requiring only client-side modifications."
    )
    
    pdf.chapter_title("References")
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5,
        "[1] McMahan, B., et al. (2017). Communication-Efficient Learning of Deep Networks "
        "from Decentralized Data. AISTATS.\n\n"
        "[2] Zhu, L., Liu, Z., & Han, S. (2019). Deep Leakage from Gradients. NeurIPS.\n\n"
        "[3] Rebuffi, S.A., et al. (2017). iCaRL: Incremental Classifier and Representation "
        "Learning. CVPR.\n\n"
        "[4] Casado, F.E., et al. (2022). Concept Drift Detection and Adaptation for "
        "Federated and Continual Learning. Multimedia Tools and Applications.\n\n"
        "[5] Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset "
        "for Benchmarking Machine Learning Algorithms. arXiv:1708.07747."
    )
    
    pdf.ln(10)
    pdf.chapter_title("Appendix: Summary Table")
    
    widths = [60, 35, 35, 50]
    pdf.add_table_row(['Approach', 'Final Acc', 'Peak Acc', 'Notes'], widths, bold=True)
    pdf.add_table_row(['Centralized Baseline', '~88%', '~88%', 'IID only, no drift'], widths)
    pdf.add_table_row(['FedAvg (no drift)', '~78%', '~80%', 'Stable IID training'], widths)
    pdf.add_table_row(['FedAvg (with drift)', '~28%', '~74%', 'Severe forgetting'], widths)
    pdf.add_table_row(['FedAvg + Replay', '~78%', '~82%', '50 samples/class buffer'], widths)

    pdf.output(REPORT_FILE)
    print(f"✅ Report Generated: {REPORT_FILE}")

if __name__ == "__main__":
    generate_pdf()
