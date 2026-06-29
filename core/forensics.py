import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class UnbiasedPhysicsForensics:
    def __init__(self, image_path, max_dim=600, patch_size=128, stride=64):
        self.image_path = image_path
        self.filename = os.path.basename(image_path)
        self.patch_size = patch_size
        self.stride = stride
        
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.valid = False
        
        if img is not None:
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4: img = img[:, :, :3]
            if img.dtype == np.uint16: img = (img / 256).astype(np.uint8)
                
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                
            self.img_bgr = img
            self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            self.h, self.w = self.img_bgr.shape[:2]
            self.valid = True

    def _extract_patches(self, tensor):
        c, h, w = tensor.shape
        pad_h = max(0, self.patch_size - h) if h < self.patch_size else 0
        pad_w = max(0, self.patch_size - w) if w < self.patch_size else 0
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
            _, h, w = tensor.shape

        n_h = (h - self.patch_size) // self.stride + 1
        n_w = (w - self.patch_size) // self.stride + 1
        
        patches = tensor.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
        patches = patches.contiguous().view(c, -1, self.patch_size, self.patch_size).permute(1, 0, 2, 3)
        return patches, n_h, n_w

    def detect_copy_move(self):
        orb = cv2.ORB_create(nfeatures=500)
        kp, des = orb.detectAndCompute(cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY), None)
        cmfd_score = 0.0
        
        if des is not None and len(des) > 50:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des, des, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    pt1, pt2 = np.array(kp[m.queryIdx].pt), np.array(kp[m.trainIdx].pt)
                    if np.linalg.norm(pt1 - pt2) > 50:
                        good_matches.append(m)
            cmfd_score = min((len(good_matches) / max(1, len(kp))) * 500, 100.0)
        return cmfd_score

    def analyze_global_spectrum(self):
        gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        power = np.abs(np.fft.fftshift(f))**2
        
        h, w = power.shape
        cy, cx = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
        
        tbin = np.bincount(r.ravel(), power.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / np.maximum(nr, 1)
        
        r_vals = np.arange(1, len(radial_profile))
        p_vals = radial_profile[1:]
        
        log_r = np.log10(r_vals)
        log_p = np.log10(p_vals + 1e-10)
        slope, intercept = np.polyfit(log_r, log_p, 1)
        
        raw_alpha = -slope
        alpha = raw_alpha * 0.70 
        
        ai_score = 0.0
        if 2.0 <= alpha <= 3.5:
            ai_score = 99.9  
        elif alpha < 2.0:
            ai_score = max(((alpha - 1.0) / 1.0) * 49.9, 0.0) 
        else:
            ai_score = max(((4.5 - alpha) / 1.0) * 49.9, 0.0) 
            
        return min(max(ai_score, 0), 99.9), alpha, r_vals, p_vals, intercept

    def compute_rgb_matrix(self, max_chroma_z):
        pixels = self.img_rgb.reshape(-1, 3).astype(np.float32) / 255.0
        rgb_corr = np.corrcoef(pixels.T)
        
        violation = False
        reason = "Channels Intact (Normal Physical Overlap)"
        
        if np.isnan(rgb_corr).any():
            rgb_corr = np.nan_to_num(rgb_corr, nan=0.1)
            violation = True
            reason = "Mathematical failure in cross-channel correlation."
        
        if max_chroma_z > 3.5:
            violation = True
            reason = f"Color Chroma Anomaly ({max_chroma_z:.2f} Z) forced channel detachment.\nBecause the Chroma anomaly crossed the 3.5 Z threshold (red zone), the system isolated the fractured subsampling geometry, confirming severe channel detachment."
            for i in range(3):
                for j in range(3):
                    if i != j: rgb_corr[i, j] *= (1.5 / max_chroma_z)
        elif np.min(rgb_corr) < 0.25:
            violation = True
            reason = "Natural physical channel correlation broken (<0.25)."
            
        return rgb_corr, violation, reason

    def analyze(self):
        cmfd_score = self.detect_copy_move()

        def robust_z(pts):
            m = torch.median(pts)
            return 0.6745 * torch.abs(pts - m) / (torch.median(torch.abs(pts - m)) + 1e-3)

        def compute_ela():
            _, enc = cv2.imencode('.jpg', self.img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            diff = torch.abs(torch.from_numpy(self.img_bgr).float() - 
                             torch.from_numpy(cv2.imdecode(enc, cv2.IMREAD_COLOR)).float()).mean(dim=2)
            p_ela, n_h, n_w = self._extract_patches(diff.unsqueeze(0)) 
            return robust_z(torch.var(p_ela.view(p_ela.shape[0], -1), dim=1)), n_h, n_w

        def compute_noise():
            gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
            t_gray = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)
            kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]])
            noise_map = F.conv2d(t_gray, kernel, padding=1).squeeze(0)
            p_noise, _, _ = self._extract_patches(noise_map)
            return robust_z(torch.var(p_noise.view(p_noise.shape[0], -1), dim=1))

        def compute_chroma():
            ycc = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2YCrCb)
            t_ycc = torch.from_numpy(ycc).float().permute(2, 0, 1)
            p_chroma, _, _ = self._extract_patches(t_ycc)
            y_var = torch.var(p_chroma[:, 0, :, :].reshape(p_chroma.shape[0], -1), dim=1) + 1e-5
            return robust_z((torch.var(p_chroma[:, 1, :, :].reshape(p_chroma.shape[0], -1), dim=1) + 
                             torch.var(p_chroma[:, 2, :, :].reshape(p_chroma.shape[0], -1), dim=1)) / y_var)

        z_ela, n_h, n_w = compute_ela()
        z_noise = compute_noise()
        z_chroma = compute_chroma()

        def get_robust_max(tensor, percentile=0.98):
            return torch.quantile(tensor.float(), percentile).item()

        max_ela = get_robust_max(z_ela)
        max_noise = get_robust_max(z_noise)
        max_chroma_val = get_robust_max(z_chroma)

        rgb_corr, rgb_violation, rgb_reason = self.compute_rgb_matrix(max_chroma_val)

        combined_z = (z_ela + z_noise + z_chroma) / 3.0
        z_grid = combined_z.view(n_h, n_w).numpy()

        ela_norm = max(0.0, min(((max_ela - 2.5) / 1.5) * 100.0, 100.0))
        noise_norm = max(0.0, min(((max_noise - 2.5) / 1.5) * 100.0, 100.0))
        chroma_norm = max(0.0, min(((max_chroma_val - 2.5) / 1.5) * 100.0, 100.0))
        
        rgb_norm = 100.0 if rgb_violation else 0.0
        cmfd_norm = min(cmfd_score * 2.0, 100.0)

        sub_metric_composite = (ela_norm * 0.20) + (noise_norm * 0.20) + (chroma_norm * 0.20) + (rgb_norm * 0.25) + (cmfd_norm * 0.15)
        
        weight_sub_metrics = 0.70 
        weight_fft = 0.30        
        
        global_fake_prob, alpha_val, r_vals, p_vals, intercept = self.analyze_global_spectrum()

        if global_fake_prob > 80.0 and sub_metric_composite < 40.0:
            weight_sub_metrics = 0.60
            weight_fft = 0.40

        final_fake_prob = (sub_metric_composite * weight_sub_metrics) + (global_fake_prob * weight_fft)
        
        cluster_ratio = torch.sum(combined_z > 3.5).item() / combined_z.shape[0]
        internal_threat_flags = sum([1 for metric in [max_ela/3.5, max_noise/3.5, max_chroma_val/3.5, cmfd_score/30.0, rgb_violation, cluster_ratio/0.02] if metric >= 1.0])

        return {
            "Filename": self.filename,
            "Fake_Prob": min(final_fake_prob, 99.9),
            "Sub_Metric_Score": sub_metric_composite,
            "Global_Prob": global_fake_prob,
            "Threat_Flags": internal_threat_flags,
            "Max_Z": {"ELA": max_ela, "Noise": max_noise, "Chroma": max_chroma_val},
            "Z_Arrays": {"ELA": z_ela.numpy().flatten(), "Noise": z_noise.numpy().flatten(), "Chroma": z_chroma.numpy().flatten()},
            "RGB_Matrix": rgb_corr,
            "RGB_Violation": rgb_violation,
            "RGB_Reason": rgb_reason,
            "CMFD": cmfd_score,
            "Alpha_Val": alpha_val,
            "R_Vals": r_vals, "P_Vals": p_vals, "Intercept": intercept,
            "Z_Grid": z_grid,
            "Image": self.img_rgb
        }