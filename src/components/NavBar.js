import LogoutIcon from "@mui/icons-material/Logout";
import { useState } from "react";
import "./NavBar.css";

function NavBar() {
  const [username, setUsername] = useState("TaiFuocVip");

  return (
    <div className="navbar">
      <div className="username-navbar">{username}</div>
      <div className="logout-block">
        <div>Logout</div>
        <LogoutIcon color="error" />
      </div>
    </div>
  );
}

export default NavBar;
